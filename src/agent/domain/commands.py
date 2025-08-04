from typing import Any, Dict, List, Optional

from pydantic import BaseModel

################################################################################
# Tools pydantic models - between agent and adapters
################################################################################


class GuardrailPreCheckModel(BaseModel):
    approved: bool
    chain_of_thought: str
    response: str


class GuardrailPostCheckModel(BaseModel):
    chain_of_thought: str
    approved: bool
    summary: str
    issues: List[str]
    plausibility: str
    factual_consistency: str
    clarity: str
    completeness: str


class KBResponse(BaseModel):
    description: str
    score: float
    id: str
    tag: str
    name: str


class LLMResponseModel(BaseModel):
    chain_of_thought: str
    response: str


class RerankResponse(BaseModel):
    question: str
    text: str
    score: float
    id: str
    tag: str
    name: str


################################################################################
# SQL building blocks
################################################################################


class AggregationFunction(BaseModel):
    function: str  # COUNT, SUM, AVG, etc.
    column: Optional[str] = None
    alias: Optional[str] = None


class ColumnMapping(BaseModel):
    question_term: str
    table_name: str
    column_name: str
    confidence: float


class FilterCondition(BaseModel):
    column: str
    operator: str  # =, >, <, LIKE, etc.
    value: str
    chain_of_thought: Optional[str] = None


class JoinPath(BaseModel):
    from_table: str
    to_table: str
    from_column: str
    to_column: str
    join_type: str = "INNER"


class TableMapping(BaseModel):
    question_term: str
    table_name: str
    confidence: float


################################################################################
# SQL pydantic models - between agent and adapters
################################################################################


class AggregationResponse(BaseModel):
    """Response from Aggregation Agent."""

    aggregations: List[AggregationFunction]
    group_by_columns: List[str] = []
    is_aggregation_query: bool
    chain_of_thought: Optional[str] = None


class ConstructionResponse(BaseModel):
    sql_query: str
    chain_of_thought: Optional[str] = None


class FilterResponse(BaseModel):
    """Response from Filter Agent."""

    conditions: List[FilterCondition]
    chain_of_thought: Optional[str] = None


class GroundingResponse(BaseModel):
    table_mapping: List[TableMapping]
    column_mapping: List[ColumnMapping]
    chain_of_thought: Optional[str] = None


class JoinInferenceResponse(BaseModel):
    joins: List[JoinPath]
    chain_of_thought: Optional[str] = None


class ValidationResponse(BaseModel):
    approved: bool
    issues: Optional[List[str]] = None
    summary: Optional[str] = None
    confidence: float
    chain_of_thought: Optional[str] = None


class Column(BaseModel):
    name: str
    type: str
    description: Optional[str] = None


class Table(BaseModel):
    name: str
    columns: list[Column]
    description: Optional[str] = None


class Relationship(BaseModel):
    table_name: str
    column_name: str
    foreign_table_name: str
    foreign_column_name: str


class DatabaseSchema(BaseModel):
    tables: list[Table]
    relationships: list[Relationship]


class ScenarioCandidate(BaseModel):
    question: str
    endpoint: str


class ScenarioResponse(BaseModel):
    candidates: List[ScenarioCandidate]
    chain_of_thought: Optional[str] = None


class ScenarioValidationResponse(BaseModel):
    approved: bool
    summary: Optional[str] = None
    issues: Optional[List[str]] = None
    plausibility: Optional[str] = None
    usefulness: Optional[str] = None
    clarity: Optional[str] = None
    chain_of_thought: Optional[str] = None


################################################################################
# Internal Tool Commands
################################################################################


class Command(BaseModel):
    pass


class Check(Command):
    question: str
    q_id: str
    approved: Optional[bool] = None
    chain_of_thought: Optional[str] = None
    response: Optional[str] = None


class Enhance(Command):
    question: str
    q_id: str
    response: Optional[str] = None
    chain_of_thought: Optional[str] = None


class FinalCheck(Command):
    question: str
    q_id: str
    chain_of_thought: Optional[str] = None
    approved: Optional[bool] = None
    summary: Optional[str] = None
    issues: Optional[List[str]] = None
    plausibility: Optional[str] = None
    factual_consistency: Optional[str] = None
    clarity: Optional[str] = None
    completeness: Optional[str] = None
    data: Optional[Dict[str, str]] = None


class LLMResponse(Command):
    question: str
    q_id: str
    response: Optional[str] = None
    data: Optional[Dict[str, str]] = None
    chain_of_thought: Optional[str] = None


class Question(Command):
    question: str
    q_id: str


class Rerank(Command):
    question: str
    q_id: str
    candidates: Optional[List[KBResponse]] = None


class Retrieve(Command):
    question: str
    q_id: str
    candidates: Optional[List[KBResponse]] = None


class UseTools(Command):
    question: str
    q_id: str
    response: Optional[str] = None
    memory: Optional[List[str]] = None
    data: Optional[Dict[str, str]] = None


################################################################################
# Internal SQL Commands
################################################################################


class SQLAggregation(Command):
    question: str
    q_id: str
    column_mapping: List[ColumnMapping]
    aggregations: Optional[List[AggregationFunction]] = None
    group_by_columns: Optional[List[str]] = None
    is_aggregation_query: Optional[bool] = None
    chain_of_thought: Optional[str] = None


class SQLCheck(Command):
    question: str
    q_id: str
    approved: Optional[bool] = None
    chain_of_thought: Optional[str] = None
    response: Optional[str] = None


class SQLConstruction(Command):
    question: str
    q_id: str
    schema_info: Optional[DatabaseSchema] = None
    column_mapping: Optional[List[ColumnMapping]] = None
    table_mapping: Optional[List[TableMapping]] = None
    conditions: Optional[List[FilterCondition]] = None
    joins: Optional[List[JoinPath]] = None
    aggregations: Optional[List[AggregationFunction]] = None
    group_by_columns: Optional[List[str]] = None
    is_aggregation_query: Optional[bool] = None
    sql_query: Optional[str] = None
    chain_of_thought: Optional[str] = None


class SQLExecution(Command):
    question: str
    q_id: str
    sql_query: str
    data: Optional[Dict[str, Any]] = None


class SQLFilter(Command):
    question: str
    q_id: str
    column_mapping: List[ColumnMapping]
    conditions: Optional[List[FilterCondition]] = None
    chain_of_thought: Optional[str] = None


class SQLGrounding(Command):
    question: str
    q_id: str
    tables: List[Table]
    table_mapping: Optional[List[TableMapping]] = None
    column_mapping: Optional[List[ColumnMapping]] = None
    chain_of_thought: Optional[str] = None


class SQLJoinInference(Command):
    question: str
    q_id: str
    table_mapping: List[TableMapping]
    relationships: List[Relationship]
    joins: Optional[List[JoinPath]] = None
    chain_of_thought: Optional[str] = None


class SQLQuestion(Command):
    question: str
    q_id: str
    schema_info: Optional[Any] = None


class SQLValidation(Command):
    question: str
    q_id: str
    sql_query: str
    tables: List[Table]
    relationships: List[Relationship]
    approved: Optional[bool] = None
    summary: Optional[str] = None
    issues: Optional[List[str]] = None
    confidence: Optional[float] = None
    chain_of_thought: Optional[str] = None


################################################################################
# Evaluation Commands
################################################################################


class StartEvaluationRun(Command):
    """Command to start a new evaluation run."""

    run_type: str
    evaluation_category: Optional[str] = None
    stage: Optional[str] = None
    model_name: Optional[str] = None
    model_temperature: Optional[float] = None
    prompt_version: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RecordTestResult(Command):
    """Command to record a test result."""

    run_id: str  # UUID as string
    test_name: str
    test_type: Optional[str] = None
    question: str
    expected_response: str
    actual_response: str
    passed: bool
    execution_time_ms: Optional[int] = None
    judge_scores: Optional[Dict[str, float]] = None
    judge_reasoning: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    # Tool agent specific
    tools_used: Optional[List[str]] = None
    tool_outputs: Optional[Dict[str, Any]] = None
    # SQL specific
    sql_stage: Optional[str] = None
    sql_query: Optional[str] = None
    schema_context: Optional[Dict[str, Any]] = None


class CompleteEvaluationRun(Command):
    """Command to complete an evaluation run."""

    run_id: str  # UUID as string
    fixtures_used: Optional[List[str]] = None


################################################################################
# Internal Scenario Commands
################################################################################


class Scenario(Command):
    question: str
    q_id: str
    schema_info: Optional[Any] = None
    tool_info: Optional[Any] = None


class ScenarioLLMResponse(Command):
    question: str
    q_id: str
    tables: Optional[List[Table]] = None
    tools: Optional[List[str]] = None
    candidates: Optional[List[ScenarioCandidate]] = None
    chain_of_thought: Optional[str] = None


class ScenarioFinalCheck(Command):
    question: str
    q_id: str
    candidates: Optional[List[ScenarioCandidate]] = None
    approved: Optional[bool] = None
    summary: Optional[str] = None
    issues: Optional[List[str]] = None
    plausibility: Optional[str] = None
    usefulness: Optional[str] = None
    clarity: Optional[str] = None
    chain_of_thought: Optional[str] = None
