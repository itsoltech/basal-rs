"""Client for the native Basal API (/v1/basal)."""

from .client import (
    AsyncBasalClient, BasalClient, BasalError, BasalHTTPError,
    BasalProtocolError, BasalTransportError, BasalResponseTooLargeError,
)
from .types import (
    Answer, DecisionRequest, DecisionResponse, EvidenceSpan,
    JsonValue, ModelMetadata, ModelsResponse, Question, Usage,
)

__all__ = [
    "AsyncBasalClient", "BasalClient", "BasalError", "BasalHTTPError",
    "BasalProtocolError", "BasalTransportError", "BasalResponseTooLargeError", "Answer", "DecisionRequest",
    "DecisionResponse", "EvidenceSpan", "JsonValue",
    "ModelMetadata", "ModelsResponse", "Question", "Usage",
]

from .questions import (
    ChoiceAnswer, NoulAnswer, ScoreAnswer, MultiAnswer, ActAnswer,
    ChoiceQuestion, NoulQuestion, ScoreQuestion, MultiQuestion, ActQuestion, QuestionSpec, QuestionSet, TypedDecisionResponse,
    Questions, SchemaDecisionResponse, AnswerResult,
    choice, noul, score, multi, act,
)
__all__ += [
    "ChoiceAnswer", "NoulAnswer", "ScoreAnswer", "MultiAnswer", "ActAnswer",
    "ChoiceQuestion", "NoulQuestion", "ScoreQuestion", "MultiQuestion", "ActQuestion", "QuestionSpec", "QuestionSet", "TypedDecisionResponse",
    "Questions", "SchemaDecisionResponse", "AnswerResult",
    "choice", "noul", "score", "multi", "act",
]

from .media import image, audio
from .types import Calibration, MediaInput, MediaUsage, MultiObjective, ToolRanking
from .questions import ToolAnswer, ToolQuestion, tool
__all__ += ["image", "audio", "Calibration", "MediaInput", "MediaUsage", "MultiObjective", "ToolRanking", "ToolAnswer", "ToolQuestion", "tool"]

from .tasks import (
    ClassifyAnswer, VerifyAnswer, CompressAnswer, ExtractAnswer,
    ClaimResult, CompressionUnit, ExtractField,
    ClassifyQuestion, VerifyQuestion, CompressQuestion, ExtractQuestion,
    classify, verify, compress, extract,
)
__all__ += [
    "ClassifyAnswer", "VerifyAnswer", "CompressAnswer", "ExtractAnswer",
    "ClaimResult", "CompressionUnit", "ExtractField",
    "ClassifyQuestion", "VerifyQuestion", "CompressQuestion", "ExtractQuestion",
    "classify", "verify", "compress", "extract",
]

from .batches import (
    AsyncUploadContent, Batch, BatchErrorEntry, BatchErrors, BatchInputLine, BatchList, BatchRecordError,
    BatchResponse, BatchResultLine, BatchStatus, CreateBatchRequest, DeletedFile, ExpiresAfter, FileList,
    FileObject, FilePurpose, RequestCounts, UploadContent,
)
__all__ += [
    "AsyncUploadContent", "Batch", "BatchErrorEntry", "BatchErrors", "BatchInputLine", "BatchList", "BatchRecordError",
    "BatchResponse", "BatchResultLine", "BatchStatus", "CreateBatchRequest", "DeletedFile", "ExpiresAfter", "FileList",
    "FileObject", "FilePurpose", "RequestCounts", "UploadContent",
]

from .batches import BatchRequest, TERMINAL_STATUSES
from .errors import BasalBatchNotReadyError, BasalBatchResultError, BasalBatchSubmissionError, BasalBatchTimeoutError
from .resources import AsyncBatches, AsyncFiles, Batches, Files
__all__ += [
    "BatchRequest", "TERMINAL_STATUSES", "BasalBatchNotReadyError", "BasalBatchResultError",
    "BasalBatchSubmissionError", "BasalBatchTimeoutError", "AsyncBatches", "AsyncFiles", "Batches", "Files",
]
