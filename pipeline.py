import numpy as np
import autokeras as ak

from pprint import pprint
from sklearn.datasets import fetch_20newsgroups


def build_multimodal_model(
    text_data=None,
    image_data=None,
    structured_data=None,
    classification_targets=None,
    regression_targets=None,
    include_text_labels=True,
    split_ratio=0.8,
    max_trials=3,
    seed=42,
    epochs=5,
):
    """
    Parameters
    ----------
    text_data : dict or None
        {
            "doc": [...],
            "label": [...]  # optional classification target
        }

    image_data : np.ndarray or None

    structured_data : np.ndarray or None

    classification_targets : list or None
        Additional classification targets.

    regression_targets : list or None
        Regression targets.

    include_text_labels : bool
        If True and text_data contains "label",
        a ClassificationHead will be created automatically.
    """

    # --------------------------------------------------
    # Determine dataset size
    # --------------------------------------------------

    if text_data is not None:
        num_instances = len(text_data["doc"])
    elif image_data is not None:
        num_instances = len(image_data)
    elif structured_data is not None:
        num_instances = len(structured_data)
    else:
        raise ValueError(
            "At least one of text_data, image_data, structured_data is required."
        )

    split_idx = int(num_instances * split_ratio)

    # --------------------------------------------------
    # Build inputs and branches
    # --------------------------------------------------

    inputs = []
    branches = []

    train_inputs = []
    test_inputs = []

    text_label_train = None
    text_label_test = None

    # --------------------------------------------------
    # TEXT
    # --------------------------------------------------

    if text_data is not None:

        docs = np.array(text_data["doc"])

        doc_train = docs[:split_idx]
        doc_test = docs[split_idx:]

        text_input = ak.TextInput()
        text_branch = ak.TextBlock()(text_input)

        inputs.append(text_input)
        branches.append(text_branch)

        train_inputs.append(doc_train)
        test_inputs.append(doc_test)

        # Optional text labels
        if include_text_labels and "label" in text_data:

            labels = np.array(text_data["label"])

            text_label_train = labels[:split_idx]
            text_label_test = labels[split_idx:]

    # --------------------------------------------------
    # IMAGE
    # --------------------------------------------------

    if image_data is not None:

        image_train = image_data[:split_idx]
        image_test = image_data[split_idx:]

        image_input = ak.ImageInput()

        image_branch = ak.Normalization()(image_input)
        image_branch = ak.ConvBlock()(image_branch)

        inputs.append(image_input)
        branches.append(image_branch)

        train_inputs.append(image_train)
        test_inputs.append(image_test)

    # --------------------------------------------------
    # STRUCTURED
    # --------------------------------------------------

    if structured_data is not None:

        structured_train = structured_data[:split_idx]
        structured_test = structured_data[split_idx:]

        structured_input = ak.StructuredDataInput()

        structured_branch = ak.StructuredDataBlock()(structured_input)
        structured_branch = ak.DenseBlock()(structured_branch)

        inputs.append(structured_input)
        branches.append(structured_branch)

        train_inputs.append(structured_train)
        test_inputs.append(structured_test)

    # --------------------------------------------------
    # MERGE
    # --------------------------------------------------

    if len(branches) == 1:
        merged = branches[0]
    else:
        merged = ak.Merge()(branches)

    # --------------------------------------------------
    # OUTPUTS
    # --------------------------------------------------

    outputs = []

    train_targets = []
    test_targets = []

    # --------------------------------------------------
    # TEXT LABEL HEAD
    # --------------------------------------------------

    if text_label_train is not None:

        train_targets.append(text_label_train)
        test_targets.append(text_label_test)

        outputs.append(
            ak.ClassificationHead(
                name="text_label"
            )(merged)
        )

    # --------------------------------------------------
    # ADDITIONAL CLASSIFICATION OUTPUTS
    # --------------------------------------------------

    if classification_targets is not None:

        for i, target in enumerate(classification_targets):

            target = np.array(target)

            train_targets.append(target[:split_idx])
            test_targets.append(target[split_idx:])

            outputs.append(
                ak.ClassificationHead(
                    name=f"classification_{i}"
                )(merged)
            )

    # --------------------------------------------------
    # REGRESSION OUTPUTS
    # --------------------------------------------------

    if regression_targets is not None:

        for i, target in enumerate(regression_targets):

            target = np.array(target)

            train_targets.append(target[:split_idx])
            test_targets.append(target[split_idx:])

            outputs.append(
                ak.RegressionHead(
                    name=f"regression_{i}"
                )(merged)
            )

    if len(outputs) == 0:
        raise ValueError(
            "At least one classification or regression target is required."
        )

    # --------------------------------------------------
    # AUTOMODEL
    # --------------------------------------------------

    auto_model = ak.AutoModel(
        inputs=inputs,
        outputs=outputs,
        max_trials=max_trials,
        overwrite=True,
        seed=seed,
    )

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    auto_model.fit(
        train_inputs,
        train_targets,
        epochs=epochs,
    )

    auto_model.tuner.results_summary()

    # --------------------------------------------------
    # EVALUATE
    # --------------------------------------------------

    results = auto_model.evaluate(
        test_inputs,
        test_targets,
        return_dict=True,
    )

    return auto_model, results


# ======================================================
# Example usage
# ======================================================

if __name__ == "__main__":

    num_instances = 1000

    categories = [
        "rec.autos",
        "rec.motorcycles",
    ]

    news = fetch_20newsgroups(
        subset="train",
        shuffle=True,
        random_state=42,
        categories=categories,
    )

    docs = np.array(news.data)[:num_instances]
    labels = np.array(news.target)[:num_instances]

    # Image data
    image_data = np.random.rand(
        num_instances,
        32,
        32,
        3,
    ).astype(np.float32)

    # Structured data
    structured_data = np.random.rand(
        num_instances,
        10,
    ).astype(np.float32)

    # Additional classification target
    classification_target = np.random.randint(
        5,
        size=num_instances,
    )

    # Regression target
    regression_target = np.random.rand(
        num_instances,
        1,
    ).astype(np.float32)

    best_model, results = build_multimodal_model(
        text_data={
            "doc": docs,
            "label": labels,  # automatically becomes a ClassificationHead
        },
        image_data=image_data,
        structured_data=structured_data,
        classification_targets=[
            classification_target,
        ],
        regression_targets=[
            regression_target,
        ],
        include_text_labels=True,
        split_ratio=0.8,
        max_trials=3,
        seed=42,
        epochs=1,
    )

    pprint(best_model)
    pprint(results)
