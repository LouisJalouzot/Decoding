from pathlib import Path

config_path = Path(__file__).parent / "configs.txt"
config_path.unlink(missing_ok=True)


def get_fine_tune(subject_index, dataset):
    fine_tune = f'--fine_tune \'{{"{dataset}": ["'
    fine_tune += f"UTS0{subject_index}"
    fine_tune += "']}'"

    return fine_tune


def get_subjects(subjects_index, dataset):
    subjects = f'--subjects \'{{"{dataset}": ["'
    subjects_index = [f"UTS0{i}" for i in subjects_index]
    subjects += '", "'.join(subjects_index)
    subjects += "\"]}'"

    return subjects


def get_tags(top_encoding_voxels=None, **kwargs):
    tags = ["--tags"]
    if top_encoding_voxels is not None:
        tags.append(f"'n_voxels={top_encoding_voxels // 1000}k'")
    # Iterate over kwargs
    for k, v in kwargs.items():
        if v is not None:
            tags.append(f"'{k}={v}'")

    return " ".join(tags)


def write_config(*args, **kwargs):
    args = list(args)
    for k, v in kwargs.items():
        if v is not None:
            args.append(f"--{k} {v}")

    return " ".join(args) + "\n"


n_configs = 0
default_config = "--return_tables --cache"
with config_path.open("w") as config_file:
    for top_encoding_voxels in [3000, 80000]:
        for dataset in ["lebel2023", "lebel2023_balanced"]:
            # Single subject
            for i in range(1, 9):
                subjects = get_subjects([i], dataset)
                for n_folds in [None, 5]:
                    tags = get_tags(
                        setup=(
                            "Single subject"
                            + (" CV" if n_folds is not None else "")
                        ),
                        dataset=dataset,
                        subject=i,
                        top_encoding_voxels=top_encoding_voxels,
                        n_folds=n_folds,
                    )
                    config_file.write(
                        write_config(
                            subjects,
                            tags,
                            default_config,
                            n_folds=n_folds,
                            top_encoding_voxels=top_encoding_voxels,
                        )
                    )
                    n_configs += 1

            # Vanilla multi subject
            for multi_subject_mode in ["individual", "shared"]:
                subjects = get_subjects(range(1, 9), dataset)
                tags = get_tags(
                    setup="Multi subject",
                    dataset=dataset,
                    multi_subject_mode=multi_subject_mode,
                    top_encoding_voxels=top_encoding_voxels,
                )
                config_file.write(
                    write_config(
                        subjects,
                        tags,
                        default_config,
                        multi_subject_mode=multi_subject_mode,
                        top_encoding_voxels=top_encoding_voxels,
                    )
                )
                n_configs += 1

    # Multi subject CV for top 10 accuracy
    for multi_subject_mode, top_encoding_voxels in [
        ("individual", 3000),
        ("shared", 80000),
    ]:
        subjects = get_subjects(range(1, 9), dataset)
        tags = get_tags(
            setup="Multi subject CV",
            dataset="lebel2023_balanced",
            multi_subject_mode=multi_subject_mode,
            top_encoding_voxels=top_encoding_voxels,
            n_folds=5,
        )
        config_file.write(
            write_config(
                subjects,
                tags,
                default_config,
                multi_subject_mode=multi_subject_mode,
                top_encoding_voxels=top_encoding_voxels,
                n_folds=5,
            )
        )
        n_configs += 1

    # Fine tune
    for subject in range(1, 9):
        for fine_tune_disjoint in [True, False]:
            for train_subjects in [set(range(1, 4)), set(range(1, 9))]:
                for multi_subject_mode, top_encoding_voxels in [
                    ("individual", 3000),
                    ("shared", 80000),
                ]:
                    if len(train_subjects) == 8 and fine_tune_disjoint:
                        continue
                    for fine_tune_whole in [True, False]:
                        subjects = get_subjects(
                            train_subjects | {subject}, "lebel2023"
                        )
                        fine_tune = get_fine_tune(subject, "lebel2023")
                        tags = get_tags(
                            setup="Fine tune",
                            dataset="lebel2023",
                            subject=subject,
                            train_subjects=(
                                "3" if len(train_subjects) < 8 else "8"
                            ),
                            fine_tune_disjoint=fine_tune_disjoint,
                            fine_tune_whole=fine_tune_whole,
                            multi_subject_mode=multi_subject_mode,
                            top_encoding_voxels=top_encoding_voxels,
                        )
                        config_file.write(
                            write_config(
                                subjects,
                                fine_tune,
                                tags,
                                default_config,
                                fine_tune_disjoint=fine_tune_disjoint,
                                fine_tune_whole=fine_tune_whole,
                                multi_subject_mode=multi_subject_mode,
                                top_encoding_voxels=top_encoding_voxels,
                            )
                        )
                        n_configs += 1

print("Number of configurations:", n_configs)
