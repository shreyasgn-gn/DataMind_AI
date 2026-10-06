from typing import Callable, Any


def classify_error(error: Exception) -> str:
    """
    Classify common execution errors into recovery categories.
    """

    message = str(error).lower()

    if (
        "categorical" in message
        or "could not convert string" in message
        or "object" in message
    ):
        return "categorical_features"

    if (
        "missing" in message
        or "nan" in message
        or "null" in message
    ):
        return "missing_values"

    if (
        "target" in message
        and (
            "not found" in message
            or "does not exist" in message
        )
    ):
        return "invalid_target"

    if (
        "feature" in message
        and "match" in message
    ):
        return "feature_mismatch"

    if (
        "empty" in message
        or "no usable" in message
    ):
        return "empty_features"

    if (
        "memory" in message
        or "allocation" in message
    ):
        return "resource_limit"

    return "unknown"


def get_recovery_action(
    error_category: str,
) -> dict:
    """
    Map an error category to a deterministic recovery action.
    """

    actions = {
        "categorical_features": {
            "action": "encode_categorical_features",
            "description": (
                "Encode categorical features before "
                "retrying model training."
            ),
        },

        "missing_values": {
            "action": "handle_missing_values",
            "description": (
                "Fill or otherwise handle missing values "
                "before retrying."
            ),
        },

        "invalid_target": {
            "action": "stop_invalid_target",
            "description": (
                "Stop execution because the requested "
                "target is invalid."
            ),
        },

        "feature_mismatch": {
            "action": "rebuild_features",
            "description": (
                "Rebuild the feature matrix so that model "
                "inputs match the training schema."
            ),
        },

        "empty_features": {
            "action": "stop_no_features",
            "description": (
                "Stop execution because no usable features "
                "remain."
            ),
        },

        "resource_limit": {
            "action": "reduce_workload",
            "description": (
                "Reduce model or dataset workload and retry."
            ),
        },

        "unknown": {
            "action": "retry_once",
            "description": (
                "Retry the operation once before reporting "
                "the failure."
            ),
        },
    }

    return actions.get(
        error_category,
        actions["unknown"],
    )


def execute_with_recovery(
    operation: Callable[[], Any],
    max_retries: int = 1,
) -> dict:
    """
    Execute an operation and retry after classifying failures.

    This function does not silently claim success. A retry is
    recorded explicitly, and unrecoverable errors are returned.
    """

    attempts = 0
    errors = []
    recovery_actions = []

    while attempts <= max_retries:

        try:

            result = operation()

            return {
                "success": True,
                "result": result,
                "attempts": attempts + 1,
                "errors": errors,
                "recovery_actions": recovery_actions,
            }

        except Exception as error:

            attempts += 1

            error_category = classify_error(
                error
            )

            recovery = get_recovery_action(
                error_category
            )

            errors.append({
                "attempt": attempts,
                "error": str(error),
                "category": error_category,
            })

            recovery_actions.append({
                "attempt": attempts,
                "action": recovery["action"],
                "description": recovery[
                    "description"
                ],
            })

            # Invalid target and no-feature errors should
            # not be retried blindly.
            if error_category in {
                "invalid_target",
                "empty_features",
            }:
                break

            if attempts > max_retries:
                break

    return {
        "success": False,
        "result": None,
        "attempts": attempts,
        "errors": errors,
        "recovery_actions": recovery_actions,
    }