"""Placeholder action schema with no robot connection."""


class RobotTools:
    def schemas(self):
        return [{"type": "function", "function": {
            "name": "move_object", "description": "Move an object to a destination.",
            "parameters": {"type": "object", "properties": {
                "object": {"type": "string"}, "destination": {"type": "string"}},
                "required": ["object", "destination"], "additionalProperties": False}}}]

    def execute(self, call):
        return {"ok": True, "note": "stub"}
