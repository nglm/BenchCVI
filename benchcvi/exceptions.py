class InconsistentLogError(Exception):
	"""Raised when a log references files inconsistent with the current state."""
	pass

class ConfigurationError(Exception):
	"""Raised when a configuration doesn't match the requirement"""