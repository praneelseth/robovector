"""Enforce explicitly reviewed condition scope before model loading."""


def validate_task_scope(tasks, conditions):
    for task in tasks:
        if 'eligible_conditions' not in task:
            continue  # Preserve the established scope of historical datasets.
        allowed = task['eligible_conditions']
        if not isinstance(allowed, list) or not allowed or any(not isinstance(c, str) for c in allowed):
            raise ValueError('Invalid eligible_conditions for ' + task['task_id'])
        missing = set(conditions) - set(allowed)
        if missing:
            raise ValueError(f"Task {task['task_id']} is not approved for conditions {sorted(missing)}")
