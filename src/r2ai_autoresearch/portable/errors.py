"""Explicit data-construction failures; infrastructure failures still abort."""
class CandidateDataError(ValueError):
    pass

class ResponseValidationError(CandidateDataError):
    pass

class CandidateEvaluationError(ValueError):
    def __init__(self, message, *, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics
