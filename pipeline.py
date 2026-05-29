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
            "label": [...]
        }

    image_data : np.ndarray or None

    structured_data : np.ndarray or None

    classification_targets : list or None
        List of classification target arrays.

        Example:
        [
            label,
            classification_target
        ]

    regression_targets : list or None
        List of regression target arrays.

        Example:
        [
            regression_target
        ]
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

    # Classification outputs

    if classification_targets is not None:

        for target in classification_targets:

            target = np.array(target)

            train_targets.append(target[:split_idx])
            test_targets.append(target[split_idx:])

            outputs.append(
                ak.ClassificationHead()(merged)
            )

    # Regression outputs

    if regression_targets is not None:

        for target in regression_targets:

            target = np.array(target)

            train_targets.append(target[:split_idx])
            test_targets.append(target[split_idx:])

            outputs.append(
                ak.RegressionHead()(merged)
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

    image_data = np.random.rand(
        num_instances,
        32,
        32,
        3,
    ).astype(np.float32)

    structured_data = np.random.rand(
        num_instances,
        10,
    ).astype(np.float32)

    classification_target = np.random.randint(
        5,
        size=num_instances,
    )

    regression_target = np.random.rand(
        num_instances,
        1,
    ).astype(np.float32)

    best_model, results = build_multimodal_model(
        text_data={
            "doc": docs,
            "label": labels,  # optional metadata only
        },
        image_data=image_data,
        structured_data=structured_data,
        classification_targets=[
            labels,
            classification_target,
        ],
        regression_targets=[
            regression_target,
        ],
        split_ratio=0.8,
        max_trials=3,
        seed=42,
        epochs=1,
    )
    pprint(best_model)
    pprint(results)