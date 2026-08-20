from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from pxr.config import load_config
from pxr.data.cohort import (
    CohortBuildError,
    apply_uncertain_policy,
    build_cohort,
    canonical_image_key,
    verify_manifest,
)
from pxr.data.contracts import validate_primary_cohort

REAL = "config/study_config.yaml"
CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices", "No Finding",
]


@pytest.fixture
def cfg():
    return load_config(REAL)


@pytest.fixture
def cfg_mimic(tmp_path):
    """Config supplying MIMIC-IV patients.csv, which the CXR release lacks."""
    data = yaml.safe_load(open(REAL, encoding="utf-8"))
    data["sites"]["mimic-cxr"]["metadata_files"]["patients"] = "patients.csv"
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_config(p)


# --------------------------- synthetic metadata ---------------------------- #


def write_nih(d, rows):
    pd.DataFrame(rows).to_csv(d / "Data_Entry_2017.csv", index=False)


def nih_row(idx, pid, findings="No Finding", view="PA", sex="M", age=50):
    return {
        "Image Index": f"{idx}.png", "Finding Labels": findings, "Follow-up #": 0,
        "Patient ID": pid, "Patient Age": age, "Patient Gender": sex, "View Position": view,
    }


def write_chexpert(d, rows, labels=None, label_name="findings_fixed.json"):
    """CheXpert Plus layout: metadata CSV + separate CheXbert JSONL labels."""
    pd.DataFrame(rows).to_csv(d / "df_chexpert_plus_240401.csv", index=False)
    lab_dir = d / "chexbert_labels"
    lab_dir.mkdir(exist_ok=True)
    if labels is None:
        labels = [chexpert_label(r["path_to_image"]) for r in rows]
    with open(lab_dir / label_name, "w") as fh:
        for rec in labels:
            fh.write(json.dumps(rec) + "\n")


def chexpert_row(pid, study=1, view="AP", frontal="Frontal", sex="Female", age=60,
                 partition="train"):
    return {
        "path_to_image": f"{partition}/patient{pid}/study{study}/view1_frontal.jpg",
        "sex": sex, "age": age, "frontal_lateral": frontal, "ap_pa": view,
        "deid_patient_id": f"patient{pid}", "split": partition,
    }


def chexpert_label(path, **positives):
    """A CheXbert record: null means 'not mentioned', not negative."""
    rec = {"path_to_image": path}
    rec.update({lab: None for lab in CHEXPERT_LABELS})
    for lab, val in positives.items():
        rec[lab.replace("_", " ")] = val
    return rec


def write_mimic(d, records, labels, patients):
    pd.DataFrame(records).to_csv(d / "mimic-cxr-2.0.0-metadata.csv", index=False)
    pd.DataFrame(labels).to_csv(d / "mimic-cxr-2.0.0-chexpert.csv", index=False)
    pd.DataFrame(patients).to_csv(d / "patients.csv", index=False)


def mimic_label_row(sid, stid, **positives):
    row = {"subject_id": sid, "study_id": stid}
    row.update({lab: 0.0 for lab in CHEXPERT_LABELS})
    for lab, val in positives.items():
        row[lab.replace("_", " ")] = val
    return row


# ------------------------------- NIH -------------------------------------- #


class TestNIH:
    def test_findings_expand_and_harmonise(self, cfg, tmp_path):
        write_nih(tmp_path, [nih_row(1, 1, "Cardiomegaly|Effusion"), nih_row(2, 2)])
        c = build_cohort("nih", cfg, tmp_path, image_manifest=["1.png", "2.png"]).cohort
        r1 = c[c.patient_id == "1"].iloc[0]
        assert r1["Cardiomegaly"] == 1.0
        assert r1["Pleural Effusion"] == 1.0      # harmonised from NIH "Effusion"
        assert r1["No Finding"] == 0.0
        assert c[c.patient_id == "2"].iloc[0]["No Finding"] == 1.0

    def test_sex_normalised(self, cfg, tmp_path):
        write_nih(tmp_path, [nih_row(1, 1, sex="F", view="AP")])
        c = build_cohort("nih", cfg, tmp_path, image_manifest=["1.png"]).cohort
        assert c.iloc[0]["sex"] == "Female" and c.iloc[0]["view"] == "AP"

    def test_implausible_age_excluded_and_counted(self, cfg, tmp_path):
        write_nih(tmp_path, [nih_row(1, 1, age=411), nih_row(2, 2, age=45)])
        res = build_cohort("nih", cfg, tmp_path, image_manifest=["1.png", "2.png"])
        assert set(res.cohort.patient_id) == {"2"}
        assert (res.audit.outcome == "demographics_missing").sum() == 1

    def test_unknown_finding_token_warns(self, cfg, tmp_path):
        write_nih(tmp_path, [nih_row(1, 1, "Martian Lung")])
        with pytest.warns(UserWarning, match="schema drift"):
            res = build_cohort("nih", cfg, tmp_path, image_manifest=["1.png"])
        assert any("Martian Lung" in w for w in res.warnings)

    def test_missing_metadata_column_raises(self, cfg, tmp_path):
        rows = [nih_row(1, 1)]
        del rows[0]["View Position"]
        write_nih(tmp_path, rows)
        with pytest.raises(CohortBuildError, match="View Position"):
            build_cohort("nih", cfg, tmp_path, image_manifest=["1.png"])

    def test_cohort_passes_contracts(self, cfg, tmp_path):
        rows = [
            nih_row(i, i, "Cardiomegaly" if i % 2 else "No Finding",
                    view="AP" if i % 2 else "PA", sex="M" if i % 2 else "F", age=30 + i)
            for i in range(1, 11)
        ]
        write_nih(tmp_path, rows)
        res = build_cohort("nih", cfg, tmp_path,
                           image_manifest=[f"{i}.png" for i in range(1, 11)])
        report = validate_primary_cohort(
            res.cohort, cfg.analysis_labels,
            observation_labels=[cfg.harmonisation("nih").get(x, x)
                                for x in cfg.observation_schema("nih")],
            expected_site="nih", expected_config_hash=cfg.cohort_hash,
            min_positives=0, strict=False,
        )
        assert report.ok, report.summary()


# ----------------------------- CheXpert ----------------------------------- #


class TestCheXpert:
    """CheXpert Plus: labels arrive from a JSONL file, not the metadata CSV."""

    def test_identifiers_parsed_from_path(self, cfg, tmp_path):
        write_chexpert(tmp_path, [chexpert_row("00001")])
        c = build_cohort("chexpert", cfg, tmp_path,
                         image_manifest=["patient00001/study1/view1_frontal.jpg"]).cohort
        assert c.iloc[0]["patient_id"] == "patient00001"
        assert c.iloc[0]["study_id"] == "patient00001/study1"

    def test_labels_joined_from_jsonl(self, cfg, tmp_path):
        rows = [chexpert_row("00001"), chexpert_row("00002")]
        labels = [
            chexpert_label(rows[0]["path_to_image"], Cardiomegaly=1.0),
            chexpert_label(rows[1]["path_to_image"], Pleural_Effusion=1.0),
        ]
        write_chexpert(tmp_path, rows, labels)
        c = build_cohort("chexpert", cfg, tmp_path, image_manifest=[
            "patient00001/study1/view1_frontal.jpg",
            "patient00002/study1/view1_frontal.jpg"]).cohort
        assert c.set_index("patient_id").loc["patient00001", "Cardiomegaly"] == 1.0
        assert c.set_index("patient_id").loc["patient00002", "Pleural Effusion"] == 1.0

    def test_unmentioned_label_stays_missing_not_negative(self, cfg, tmp_path):
        """null means the report did not mention it - never a negative."""
        rows = [chexpert_row("00001")]
        write_chexpert(tmp_path, rows,
                       [chexpert_label(rows[0]["path_to_image"], Cardiomegaly=1.0)])
        c = build_cohort("chexpert", cfg, tmp_path,
                         image_manifest=["patient00001/study1/view1_frontal.jpg"]).cohort
        assert pd.isna(c.iloc[0]["Pneumonia"])

    def test_unlabelled_images_counted_in_flow(self, cfg, tmp_path):
        rows = [chexpert_row("00001"), chexpert_row("00002")]
        write_chexpert(tmp_path, rows,
                       [chexpert_label(rows[0]["path_to_image"])])   # second has no record
        res = build_cohort("chexpert", cfg, tmp_path, image_manifest=[
            "patient00001/study1/view1_frontal.jpg",
            "patient00002/study1/view1_frontal.jpg"], min_manifest_match_rate=0.4)
        note = res.flow.to_frame().set_index("step").loc["labels_joined", "note"]
        assert "1 image rows had no entry" in note

    def test_missing_label_file_raises(self, cfg, tmp_path):
        pd.DataFrame([chexpert_row("00001")]).to_csv(
            tmp_path / "df_chexpert_plus_240401.csv", index=False)
        with pytest.raises(CohortBuildError, match="label file not found"):
            build_cohort("chexpert", cfg, tmp_path,
                         image_manifest=["patient00001/study1/view1_frontal.jpg"])

    def test_lateral_is_design_exclusion_not_missing_data(self, cfg, tmp_path):
        """ap_pa is blank on laterals by construction, not by omission."""
        rows = [chexpert_row("00001", frontal="Lateral", view=None),
                chexpert_row("00002", frontal="Frontal", view="PA")]
        write_chexpert(tmp_path, rows)
        res = build_cohort("chexpert", cfg, tmp_path, image_manifest=[
            "patient00001/study1/view1_frontal.jpg",
            "patient00002/study1/view1_frontal.jpg"])
        outcomes = set(res.audit.outcome)
        assert "lateral" in outcomes and "view_missing" not in outcomes

    def test_partition_read_from_split_column(self, cfg, tmp_path):
        write_chexpert(tmp_path, [chexpert_row("00001", partition="train"),
                                  chexpert_row("00002", partition="valid")])
        res = build_cohort("chexpert", cfg, tmp_path, image_manifest=[
            "patient00001/study1/view1_frontal.jpg",
            "patient00002/study1/view1_frontal.jpg"])
        assert set(res.cohort["source_partition"]) == {"train", "valid"}

    def test_unknown_sex_excluded(self, cfg, tmp_path):
        write_chexpert(tmp_path, [chexpert_row("00001", sex="Unknown"),
                                  chexpert_row("00002", sex="Male")])
        res = build_cohort("chexpert", cfg, tmp_path, image_manifest=[
            "patient00001/study1/view1_frontal.jpg",
            "patient00002/study1/view1_frontal.jpg"])
        assert set(res.cohort.patient_id) == {"patient00002"}
        assert (res.audit.outcome == "demographics_missing").sum() == 1

    def test_uncertain_policies(self, cfg, tmp_path):
        rows = [chexpert_row("00001")]
        write_chexpert(tmp_path, rows,
                       [chexpert_label(rows[0]["path_to_image"], Cardiomegaly=-1.0)])
        m = ["patient00001/study1/view1_frontal.jpg"]
        assert pd.isna(build_cohort("chexpert", cfg, tmp_path,
                                    image_manifest=m).cohort.iloc[0]["Cardiomegaly"])
        assert build_cohort("chexpert", cfg, tmp_path, image_manifest=m,
                            uncertain_policy="u_zeros").cohort.iloc[0]["Cardiomegaly"] == 0.0

    def test_malformed_path_raises(self, cfg, tmp_path):
        rows = [chexpert_row("00001")]
        rows[0]["path_to_image"] = "train/badly_named.jpg"
        write_chexpert(tmp_path, rows)
        with pytest.raises(CohortBuildError, match="image paths"):
            build_cohort("chexpert", cfg, tmp_path, image_manifest=["x/y/badly_named.jpg"])


# ------------------------------- MIMIC ------------------------------------ #


class TestMIMIC:
    def _write(self, d):
        write_mimic(
            d,
            records=[
                {"dicom_id": "i1", "subject_id": 10, "study_id": 100,
                 "ViewPosition": "AP", "StudyDate": 20160101},
                {"dicom_id": "i2", "subject_id": 11, "study_id": 200,
                 "ViewPosition": "LATERAL", "StudyDate": 20150101},
                {"dicom_id": "i3", "subject_id": 12, "study_id": 300,
                 "ViewPosition": "PA", "StudyDate": 20150101},
                {"dicom_id": "i4", "subject_id": 13, "study_id": 400,
                 "ViewPosition": "PA", "StudyDate": 20150101},
            ],
            labels=[
                mimic_label_row(10, 100, Cardiomegaly=1.0),
                mimic_label_row(11, 200, No_Finding=1.0),
                mimic_label_row(12, 300, No_Finding=1.0),
                # subject 13 / study 400 deliberately unlabelled -> join attrition
            ],
            patients=[
                {"subject_id": s, "gender": "F", "anchor_age": 50, "anchor_year": 2015}
                for s in (10, 11, 12, 13)
            ],
        )

    def test_unconfigured_patients_file_raises(self, tmp_path):
        """MIMIC-CXR carries no age or sex; without MIMIC-IV the build must stop."""
        data = yaml.safe_load(open(REAL, encoding="utf-8"))
        data["sites"]["mimic-cxr"]["metadata_files"]["patients"] = None
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(data), encoding="utf-8")
        cfg_no_demo = load_config(p)
        self._write(tmp_path)
        with pytest.raises(CohortBuildError, match="patients.csv"):
            build_cohort("mimic-cxr", cfg_no_demo, tmp_path, image_manifest=["i1"])

    def test_labels_come_from_the_images_own_study(self, cfg_mimic, tmp_path):
        """A patient must not inherit labels pooled across their other studies."""
        write_mimic(
            tmp_path,
            records=[
                {"dicom_id": "a", "subject_id": 10, "study_id": 100,
                 "ViewPosition": "AP", "StudyDate": 20150101},
                {"dicom_id": "b", "subject_id": 20, "study_id": 200,
                 "ViewPosition": "PA", "StudyDate": 20160101},
            ],
            labels=[mimic_label_row(10, 100, Cardiomegaly=1.0),
                    mimic_label_row(20, 200, No_Finding=1.0)],
            patients=[{"subject_id": s, "gender": "M", "anchor_age": 60, "anchor_year": 2015}
                      for s in (10, 20)],
        )
        c = build_cohort("mimic-cxr", cfg_mimic, tmp_path, image_manifest=["a", "b"]).cohort
        a = c[c.image_id == "a"].iloc[0]
        assert a["Cardiomegaly"] == 1.0 and a["No Finding"] == 0.0
        b = c[c.image_id == "b"].iloc[0]
        assert b["No Finding"] == 1.0 and b["Cardiomegaly"] == 0.0

    def test_age_derived_from_anchor_and_study_year(self, cfg_mimic, tmp_path):
        self._write(tmp_path)
        c = build_cohort("mimic-cxr", cfg_mimic, tmp_path, image_manifest=["i1"]).cohort
        assert c.iloc[0]["age"] == 51.0     # anchor 50 in 2015, study in 2016

    def test_join_attrition_recorded_with_projection_breakdown(self, cfg_mimic, tmp_path):
        self._write(tmp_path)
        res = build_cohort("mimic-cxr", cfg_mimic, tmp_path,
                           image_manifest=["i1", "i2", "i3"], min_manifest_match_rate=0.5)
        note = res.flow.to_frame().set_index("step").loc["labels_joined", "note"]
        assert "1 image rows had no study-level label" in note
        assert "PA" in note

    def test_lateral_excluded_and_counted_separately(self, cfg_mimic, tmp_path):
        self._write(tmp_path)
        res = build_cohort("mimic-cxr", cfg_mimic, tmp_path,
                           image_manifest=["i1", "i2", "i3"], min_manifest_match_rate=0.5)
        assert set(res.cohort.image_id) == {"i1", "i3"}
        summary = res.exclusion_summary().set_index("outcome")["n_images"]
        assert summary["lateral"] == 1

    def test_nonstandard_projection_separated_from_missing(self, cfg_mimic, tmp_path):
        write_mimic(
            tmp_path,
            records=[
                {"dicom_id": "x", "subject_id": 10, "study_id": 100,
                 "ViewPosition": "AP AXIAL", "StudyDate": 20150101},
                {"dicom_id": "y", "subject_id": 11, "study_id": 200,
                 "ViewPosition": None, "StudyDate": 20150101},
            ],
            labels=[mimic_label_row(10, 100), mimic_label_row(11, 200)],
            patients=[{"subject_id": s, "gender": "M", "anchor_age": 60, "anchor_year": 2015}
                      for s in (10, 11)],
        )
        res = build_cohort("mimic-cxr", cfg_mimic, tmp_path, image_manifest=["x", "y"])
        summary = res.exclusion_summary().set_index("outcome")["n_images"]
        assert summary["view_nonstandard"] == 1
        assert summary["view_missing"] == 1
        assert res.cohort.empty


# --------------------------- manifest matching ---------------------------- #


class TestManifest:
    def test_canonical_key_strips_path_and_extension(self):
        keys = canonical_image_key(pd.Series(["a/b/c.jpg", "d.PNG", "plain"]))
        assert list(keys) == ["c", "d", "plain"]

    def test_manifest_restricts_cohort_to_held_images(self, cfg, tmp_path):
        write_nih(tmp_path, [nih_row(i, i) for i in range(1, 6)])
        res = build_cohort("nih", cfg, tmp_path, image_manifest=["2.png", "4.png"])
        assert set(res.cohort.image_id) == {"2.png", "4.png"}

    def test_extension_mismatch_still_matches(self, cfg, tmp_path):
        """Held names without extensions must match metadata filenames."""
        write_nih(tmp_path, [nih_row(1, 1)])
        res = build_cohort("nih", cfg, tmp_path, image_manifest=["1"])
        assert len(res.cohort) == 1

    def test_low_match_rate_raises(self, cfg, tmp_path):
        write_nih(tmp_path, [nih_row(1, 1)])
        with pytest.raises(CohortBuildError, match="matched a metadata row"):
            build_cohort("nih", cfg, tmp_path,
                         image_manifest=[f"absent{i}.png" for i in range(19)] + ["1.png"])

    def test_empty_manifest_raises(self, cfg, tmp_path):
        write_nih(tmp_path, [nih_row(1, 1)])
        with pytest.raises(CohortBuildError, match="empty"):
            build_cohort("nih", cfg, tmp_path, image_manifest=[])

    def test_absent_manifest_warns(self, cfg, tmp_path):
        write_nih(tmp_path, [nih_row(1, 1)])
        with pytest.warns(UserWarning, match="defined by metadata"):
            res = build_cohort("nih", cfg, tmp_path)
        assert any("image_manifest" in w for w in res.warnings)


# ------------------------ integrity and reporting ------------------------- #


class TestIntegrityAndReporting:
    def test_duplicate_patient_raises_rather_than_selecting(self, cfg, tmp_path):
        """Selection happened upstream; this build must not silently choose."""
        write_nih(tmp_path, [nih_row(1, 7), nih_row(2, 7)])
        with pytest.raises(CohortBuildError, match="more than one retained image"):
            build_cohort("nih", cfg, tmp_path, image_manifest=["1.png", "2.png"])

    def test_flow_reports_every_stage(self, cfg, tmp_path):
        write_nih(tmp_path, [nih_row(i, i) for i in range(1, 6)])
        res = build_cohort("nih", cfg, tmp_path,
                           image_manifest=[f"{i}.png" for i in range(1, 6)])
        steps = set(res.flow.to_frame().step)
        assert {"metadata_rows", "manifest_matched", "retained", "final"} <= steps

    def test_audit_retains_rows_the_cohort_drops(self, cfg, tmp_path):
        write_nih(tmp_path, [nih_row(1, 1, age=411), nih_row(2, 2, age=50)])
        res = build_cohort("nih", cfg, tmp_path, image_manifest=["1.png", "2.png"])
        assert len(res.audit) == 2 and len(res.cohort) == 1
        assert set(res.audit.columns) >= {"sex", "age", "view_raw", "outcome", "site"}

    def test_provenance_stamped(self, cfg, tmp_path):
        write_nih(tmp_path, [nih_row(1, 1)])
        c = build_cohort("nih", cfg, tmp_path, image_manifest=["1.png"]).cohort
        assert set(c.site) == {"nih"} and set(c.config_hash) == {cfg.cohort_hash}


class TestNoFindingDerivation:
    """No Finding is derived from observed pathologies, excluding equipment."""

    def test_derived_zero_when_a_pathology_is_positive(self, cfg, tmp_path):
        """Derivation overrides the labeller, which contradicts itself."""
        rows = [chexpert_row("00001")]
        write_chexpert(tmp_path, rows,
                       [chexpert_label(rows[0]["path_to_image"], Cardiomegaly=1.0,
                                       No_Finding=1.0)])
        c = build_cohort("chexpert", cfg, tmp_path,
                         image_manifest=["patient00001/study1/view1_frontal.jpg"]).cohort
        assert c.iloc[0]["No Finding"] == 0.0

    def test_derived_one_when_no_pathology_is_positive(self, cfg, tmp_path):
        rows = [chexpert_row("00001")]
        write_chexpert(tmp_path, rows, [chexpert_label(rows[0]["path_to_image"])])
        c = build_cohort("chexpert", cfg, tmp_path,
                         image_manifest=["patient00001/study1/view1_frontal.jpg"]).cohort
        assert c.iloc[0]["No Finding"] == 1.0

    def test_equipment_alone_does_not_make_a_radiograph_abnormal(self, cfg, tmp_path):
        """A support device is hardware, not a finding."""
        rows = [chexpert_row("00001")]
        write_chexpert(tmp_path, rows,
                       [chexpert_label(rows[0]["path_to_image"], Support_Devices=1.0)])
        c = build_cohort("chexpert", cfg, tmp_path,
                         image_manifest=["patient00001/study1/view1_frontal.jpg"]).cohort
        assert c.iloc[0]["No Finding"] == 1.0

    def test_unmodelled_pathology_still_blocks_no_finding(self, cfg, tmp_path):
        """Lung Opacity is observed but not analysed - the image is still abnormal."""
        rows = [chexpert_row("00001")]
        write_chexpert(tmp_path, rows,
                       [chexpert_label(rows[0]["path_to_image"], Lung_Opacity=1.0)])
        c = build_cohort("chexpert", cfg, tmp_path,
                         image_manifest=["patient00001/study1/view1_frontal.jpg"]).cohort
        assert c.iloc[0]["No Finding"] == 0.0


class TestHarmonisationAndBins:
    def test_many_to_one_harmonisation_rejected(self, tmp_path):
        data = yaml.safe_load(open(REAL, encoding="utf-8"))
        data["labels"]["harmonisation"]["nih14"] = {
            "Effusion": "Pleural Effusion", "Infiltration": "Pleural Effusion",
        }
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(data), encoding="utf-8")
        cfg2 = load_config(p)
        write_nih(tmp_path, [nih_row(1, 1)])
        with pytest.raises(CohortBuildError, match="same target"):
            build_cohort("nih", cfg2, tmp_path, image_manifest=["1.png"])

    def test_bins_not_covering_eligible_ages_rejected_at_config_layer(self, tmp_path):
        """An eligible age with no bin would vanish from age-stratified analyses."""
        from pxr.config import ConfigError

        data = yaml.safe_load(open(REAL, encoding="utf-8"))
        data["demographics"]["age_bins"] = {"edges": [18, 40, 60], "labels": ["18-39", "40-59"]}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ConfigError, match="cohort.age_max"):
            load_config(p)

    def test_unknown_uncertain_policy_raises(self):
        with pytest.raises(CohortBuildError, match="uncertain_policy"):
            apply_uncertain_policy(pd.DataFrame({"A": [1.0]}), ["A"], "invent")

    def test_only_minus_one_is_altered(self):
        out = apply_uncertain_policy(pd.DataFrame({"A": [1.0, 0.0, -1.0, None]}), ["A"], "u_zeros")
        assert list(out.A[:3]) == [1.0, 0.0, 0.0] and pd.isna(out.A[3])


class TestImageKeyDepth:
    def test_depth_keeps_path_components(self):
        keys = canonical_image_key(pd.Series(["root/patient1/study1/view1.jpg"]), depth=3)
        assert list(keys) == ["patient1/study1/view1"]

    def test_chexpert_filenames_collapse_at_depth_one(self):
        """Why CheXpert declares depth 3: the filename alone is not unique."""
        paths = pd.Series([
            "CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg",
            "CheXpert-v1.0/train/patient00002/study1/view1_frontal.jpg",
        ])
        assert canonical_image_key(paths, depth=1).nunique() == 1
        assert canonical_image_key(paths, depth=3).nunique() == 2

    def test_backslash_paths_normalised(self):
        keys = canonical_image_key(pd.Series([r"a\\b\\c.jpg"]), depth=1)
        assert list(keys) == ["c"]

    def test_zero_depth_rejected(self):
        with pytest.raises(CohortBuildError, match="depth"):
            canonical_image_key(pd.Series(["a.jpg"]), depth=0)


class TestPatternKeys:
    """Held image names are flattened; metadata keeps directories. Keys must bridge that."""

    PATTERN = r"patient(\d+)[/_]study(\d+)[/_](view\d+_[a-zA-Z]+)"

    def test_flattened_and_nested_names_produce_the_same_key(self):
        flat = pd.Series(["train_patient40201_study1_view1_frontal.jpg"])
        nested = pd.Series(["CheXpert-v1.0/train/patient40201/study1/view1_frontal.jpg"])
        assert (
            canonical_image_key(flat, pattern=self.PATTERN).iloc[0]
            == canonical_image_key(nested, pattern=self.PATTERN).iloc[0]
        )

    def test_pattern_keys_stay_unique_across_patients_and_studies(self):
        names = pd.Series([
            "train_patient1_study1_view1_frontal.jpg",
            "train_patient1_study2_view1_frontal.jpg",
            "train_patient2_study1_view1_frontal.jpg",
        ])
        assert canonical_image_key(names, pattern=self.PATTERN).nunique() == 3

    def test_unmatched_manifest_names_raise(self, cfg, tmp_path):
        """A pattern that does not fit the naming must fail loudly, not silently drop."""
        data = yaml.safe_load(open(REAL, encoding="utf-8"))
        data["sites"]["nih"]["image_key_pattern"] = self.PATTERN
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(data), encoding="utf-8")
        cfg2 = load_config(p)
        write_nih(tmp_path, [nih_row(1, 1)])
        with pytest.raises(CohortBuildError, match="image_key_pattern"):
            build_cohort("nih", cfg2, tmp_path, image_manifest=["1.png"])

    def test_extension_and_case_differences_absorbed(self):
        a = canonical_image_key(pd.Series(["00015868_000.jpg"]))
        b = canonical_image_key(pd.Series(["00015868_000.png"]))
        assert a.iloc[0] == b.iloc[0]


class TestVerifyManifest:
    """Pre-build reconciliation: a silent partial match is worse than a refusal."""

    def test_clean_manifest_passes_every_check(self, cfg, tmp_path):
        write_nih(tmp_path, [nih_row(i, i) for i in range(1, 6)])
        report = verify_manifest("nih", cfg, tmp_path,
                                 [f"{i}.png" for i in range(1, 6)])
        assert report.ok.all(), report.to_string(index=False)

    def test_extension_difference_does_not_fail_verification(self, cfg, tmp_path):
        write_nih(tmp_path, [nih_row(1, 1)])
        report = verify_manifest("nih", cfg, tmp_path, ["1.jpg"])   # metadata says .png
        assert report.set_index("check").loc["held_matched", "ok"]

    def test_unmatched_names_flagged_before_building(self, cfg, tmp_path):
        write_nih(tmp_path, [nih_row(1, 1)])
        report = verify_manifest("nih", cfg, tmp_path,
                                 ["1.png"] + [f"ghost{i}.png" for i in range(9)])
        row = report.set_index("check").loc["held_matched"]
        assert not row.ok and "ghost" in row.note

    def test_empty_manifest_raises(self, cfg, tmp_path):
        write_nih(tmp_path, [nih_row(1, 1)])
        with pytest.raises(CohortBuildError, match="empty"):
            verify_manifest("nih", cfg, tmp_path, [])