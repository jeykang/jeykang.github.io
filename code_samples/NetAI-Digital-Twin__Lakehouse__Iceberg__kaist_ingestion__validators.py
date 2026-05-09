"""
Data Quality Validators for KAIST E2E Dataset.

Provides validation rules for:
1. Schema conformance
2. Referential integrity
3. Data quality metrics
4. Annotation sanity checks
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    abs as spark_abs,
    array_size,
    col,
    count,
    expr,
    isnan,
    isnull,
    lit,
    pow as spark_pow,
    size,
    sqrt,
    sum as spark_sum,
    when,
)

from .config import PipelineConfig


class Severity(Enum):
    """Validation failure severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    rule_name: str
    table_name: str
    passed: bool
    severity: Severity
    message: str
    failed_count: int = 0
    total_count: int = 0
    
    @property
    def pass_rate(self) -> float:
        if self.total_count == 0:
            return 1.0
        return (self.total_count - self.failed_count) / self.total_count


@dataclass
class ValidationReport:
    """Complete validation report for a pipeline run."""
    results: List[ValidationResult] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        """Check if all critical validations passed."""
        return all(
            r.passed for r in self.results if r.severity == Severity.CRITICAL
        )
    
    @property
    def critical_failures(self) -> List[ValidationResult]:
        return [r for r in self.results if not r.passed and r.severity == Severity.CRITICAL]
    
    @property
    def warnings(self) -> List[ValidationResult]:
        return [r for r in self.results if not r.passed and r.severity == Severity.WARNING]
    
    def add(self, result: ValidationResult) -> None:
        self.results.append(result)
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "VALIDATION REPORT",
            "=" * 60,
        ]
        
        for result in self.results:
            status = "✓" if result.passed else "✗"
            lines.append(f"  {status} [{result.severity.value.upper()}] {result.table_name}.{result.rule_name}")
            if not result.passed:
                lines.append(f"      {result.message}")
                lines.append(f"      Failed: {result.failed_count}/{result.total_count} ({(1-result.pass_rate)*100:.1f}%)")
        
        lines.append("=" * 60)
        lines.append(f"CRITICAL: {len([r for r in self.results if r.severity == Severity.CRITICAL and r.passed])}/{len([r for r in self.results if r.severity == Severity.CRITICAL])} passed")
        lines.append(f"WARNINGS: {len(self.warnings)} issues")
        lines.append(f"OVERALL: {'PASSED' if self.passed else 'FAILED'}")
        
        return "\n".join(lines)


class DataValidator:
    """
    Validates KAIST dataset tables for quality and integrity.
    """
    
    def __init__(self, spark: SparkSession, config: PipelineConfig):
        self.spark = spark
        self.config = config
        self.catalog = config.spark_catalog_name
        
    def _table(self, namespace: str, table: str) -> str:
        return f"{self.catalog}.{namespace}.{table}"
    
    def _read(self, namespace: str, table: str) -> DataFrame:
        return self.spark.table(self._table(namespace, table))
    
    # =========================================================================
    # Generic Validators
    # =========================================================================
    
    def check_primary_key_unique(
        self,
        namespace: str,
        table: str,
        pk_columns: List[str],
    ) -> ValidationResult:
        """Check that primary key columns are unique."""
        df = self._read(namespace, table)
        total = df.count()
        distinct = df.select(*pk_columns).distinct().count()
        
        passed = total == distinct
        failed = total - distinct
        
        return ValidationResult(
            rule_name="pk_unique",
            table_name=table,
            passed=passed,
            severity=Severity.CRITICAL,
            message=f"Primary key columns {pk_columns} must be unique",
            failed_count=failed,
            total_count=total,
        )
    
    def check_not_null(
        self,
        namespace: str,
        table: str,
        column: str,
        severity: Severity = Severity.WARNING,
    ) -> ValidationResult:
        """Check that a column has no null values."""
        df = self._read(namespace, table)
        total = df.count()
        null_count = df.filter(col(column).isNull()).count()
        
        return ValidationResult(
            rule_name=f"{column}_not_null",
            table_name=table,
            passed=null_count == 0,
            severity=severity,
            message=f"Column {column} should not contain nulls",
            failed_count=null_count,
            total_count=total,
        )
    
    def check_foreign_key(
        self,
        namespace: str,
        table: str,
        fk_column: str,
        ref_namespace: str,
        ref_table: str,
        ref_column: str,
    ) -> ValidationResult:
        """Check that foreign key values exist in referenced table."""
        df = self._read(namespace, table)
        ref_df = self._read(ref_namespace, ref_table)
        
        total = df.count()
        
        # Find orphaned foreign keys
        orphans = df.join(
            ref_df.select(col(ref_column).alias("_ref_key")),
            df[fk_column] == col("_ref_key"),
            how="left_anti"
        ).count()
        
        return ValidationResult(
            rule_name=f"{fk_column}_fk",
            table_name=table,
            passed=orphans == 0,
            severity=Severity.CRITICAL,
            message=f"Foreign key {fk_column} references non-existent {ref_table}.{ref_column}",
            failed_count=orphans,
            total_count=total,
        )
    
    def check_array_not_empty(
        self,
        namespace: str,
        table: str,
        column: str,
        severity: Severity = Severity.WARNING,
    ) -> ValidationResult:
        """Check that array columns are not empty."""
        df = self._read(namespace, table)
        total = df.count()
        
        empty_count = df.filter(
            col(column).isNull() | (size(col(column)) == 0)
        ).count()
        
        return ValidationResult(
            rule_name=f"{column}_not_empty",
            table_name=table,
            passed=empty_count == 0,
            severity=severity,
            message=f"Array column {column} should not be empty",
            failed_count=empty_count,
            total_count=total,
        )
    
    # =========================================================================
    # Domain-Specific Validators
    # =========================================================================
    
    def check_quaternion_normalized(
        self,
        namespace: str,
        table: str,
        quat_column: str,
        tolerance: float = 0.01,
    ) -> ValidationResult:
        """Check that quaternion rotations are normalized (magnitude ≈ 1)."""
        df = self._read(namespace, table)
        total = df.count()
        
        # Compute magnitude: sqrt(qw² + qx² + qy² + qz²)
        df_with_norm = df.withColumn(
            "_quat_norm",
            sqrt(
                spark_pow(col(f"{quat_column}.qw"), 2) +
                spark_pow(col(f"{quat_column}.qx"), 2) +
                spark_pow(col(f"{quat_column}.qy"), 2) +
                spark_pow(col(f"{quat_column}.qz"), 2)
            )
        )
        
        unnormalized = df_with_norm.filter(
            spark_abs(col("_quat_norm") - 1.0) > tolerance
        ).count()
        
        return ValidationResult(
            rule_name=f"{quat_column}_normalized",
            table_name=table,
            passed=unnormalized == 0,
            severity=Severity.WARNING,
            message=f"Quaternion {quat_column} should be normalized (|q| = 1 ± {tolerance})",
            failed_count=unnormalized,
            total_count=total,
        )
    
    def check_timestamp_ordering(
        self,
        namespace: str,
        table: str,
        partition_col: str,
        order_col: str,
        timestamp_col: str,
    ) -> ValidationResult:
        """Check that timestamps are monotonically increasing within partitions."""
        # This is a simplified check - full implementation would use window functions
        df = self._read(namespace, table)
        total = df.count()
        
        # For now, just check that timestamps are positive
        negative_ts = df.filter(col(timestamp_col) < 0).count()
        
        return ValidationResult(
            rule_name=f"{timestamp_col}_valid",
            table_name=table,
            passed=negative_ts == 0,
            severity=Severity.WARNING,
            message=f"Timestamps in {timestamp_col} should be non-negative",
            failed_count=negative_ts,
            total_count=total,
        )
    
    # =========================================================================
    # Full Validation Suites
    # =========================================================================
    
    def validate_bronze_layer(self) -> ValidationReport:
        """Run all validations on Bronze layer tables."""
        report = ValidationReport()
        bronze = self.config.kaist.namespace_bronze
        
        # Session validations
        report.add(self.check_primary_key_unique(bronze, "session", ["session_id"]))
        
        # Clip validations
        report.add(self.check_primary_key_unique(bronze, "clip", ["clip_id"]))
        report.add(self.check_foreign_key(bronze, "clip", "session_id", bronze, "session", "session_id"))
        
        # Frame validations
        report.add(self.check_primary_key_unique(bronze, "frame", ["frame_id"]))
        report.add(self.check_foreign_key(bronze, "frame", "clip_id", bronze, "clip", "clip_id"))
        
        # Camera validations
        report.add(self.check_foreign_key(bronze, "camera", "frame_id", bronze, "frame", "frame_id"))
        report.add(self.check_not_null(bronze, "camera", "filename", Severity.CRITICAL))
        
        # Lidar validations
        report.add(self.check_foreign_key(bronze, "lidar", "frame_id", bronze, "frame", "frame_id"))
        report.add(self.check_not_null(bronze, "lidar", "filename", Severity.CRITICAL))
        
        # Dynamic object validations
        report.add(self.check_foreign_key(bronze, "dynamic_object", "frame_id", bronze, "frame", "frame_id"))
        report.add(self.check_array_not_empty(bronze, "dynamic_object", "boxes_3d", Severity.WARNING))
        
        return report
    
    def validate_silver_layer(self) -> ValidationReport:
        """Run all validations on Silver layer tables."""
        report = ValidationReport()
        silver = self.config.kaist.namespace_silver
        
        # Same structural checks as Bronze
        report.add(self.check_primary_key_unique(silver, "session", ["session_id"]))
        report.add(self.check_primary_key_unique(silver, "clip", ["clip_id"]))
        report.add(self.check_primary_key_unique(silver, "frame", ["frame_id"]))
        
        # Additional Silver-specific checks
        report.add(self.check_timestamp_ordering(
            silver, "camera", "clip_id", "frame_id", "sensor_timestamp"
        ))
        report.add(self.check_timestamp_ordering(
            silver, "lidar", "clip_id", "frame_id", "sensor_timestamp"
        ))
        
        # Domain-specific: quaternion normalization on ego_motion rotation
        report.add(self.check_quaternion_normalized(
            silver, "ego_motion", "rotation", tolerance=0.01
        ))
        
        return report
    
    def validate_gold_layer(self) -> ValidationReport:
        """Run all validations on Gold layer tables."""
        report = ValidationReport()
        gold = self.config.kaist.namespace_gold
        
        # Check that Gold tables are not empty
        for table in ["camera_annotations", "lidar_with_ego", "sensor_fusion_frame"]:
            try:
                df = self._read(gold, table)
                count = df.count()
                report.add(ValidationResult(
                    rule_name="not_empty",
                    table_name=table,
                    passed=count > 0,
                    severity=Severity.CRITICAL,
                    message="Gold table should not be empty",
                    failed_count=1 if count == 0 else 0,
                    total_count=1,
                ))
            except Exception as e:
                report.add(ValidationResult(
                    rule_name="exists",
                    table_name=table,
                    passed=False,
                    severity=Severity.CRITICAL,
                    message=f"Gold table does not exist or cannot be read: {e}",
                    failed_count=1,
                    total_count=1,
                ))
        
        return report


def run_validation(
    config: Optional[PipelineConfig] = None,
    layers: Optional[List[str]] = None,
) -> ValidationReport:
    """
    Run validation on specified layers.
    
    Args:
        config: Pipeline configuration
        layers: List of layers to validate ("bronze", "silver", "gold")
        
    Returns:
        Combined validation report
    """
    from .config import build_spark_session
    
    if config is None:
        config = PipelineConfig()
    if layers is None:
        layers = ["bronze", "silver", "gold"]
        
    spark = build_spark_session(config, app_name="kaist-validation")
    
    try:
        validator = DataValidator(spark, config)
        report = ValidationReport()
        
        if "bronze" in layers:
            bronze_report = validator.validate_bronze_layer()
            report.results.extend(bronze_report.results)
            
        if "silver" in layers:
            silver_report = validator.validate_silver_layer()
            report.results.extend(silver_report.results)
            
        if "gold" in layers:
            gold_report = validator.validate_gold_layer()
            report.results.extend(gold_report.results)
            
        return report
        
    finally:
        spark.stop()


if __name__ == "__main__":
    report = run_validation()
    print(report.summary())
