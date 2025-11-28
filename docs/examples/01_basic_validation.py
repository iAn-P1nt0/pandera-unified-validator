"""
Example 1: Basic Data Validation

Learn the fundamentals of pandera-unified-validator validation with a simple user dataset.
"""

import pandas as pd
from pandera_unified_validator import SchemaBuilder, UnifiedValidator, ValidationReporter


def main():
    """Demonstrate basic validation workflow."""
    print("=" * 80)
    print("Example 1: Basic Data Validation")
    print("=" * 80)
    print()

    # Step 1: Define a schema using SchemaBuilder
    print("Step 1: Creating validation schema...")
    schema = (
        SchemaBuilder("user_data")
        .add_column("user_id", int, nullable=False, unique=True, ge=1)
        .add_column("username", str, nullable=False)
        .add_column("email", str, nullable=False, pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
        .add_column("age", int, nullable=False, ge=18, le=120)
        .add_column("is_active", bool, nullable=False)
        .with_metadata(version="1.0", description="User registration schema")
        .build()
    )
    print(f"✓ Schema created with {len(schema.columns)} columns")
    print()

    # Step 2: Prepare sample data
    print("Step 2: Loading sample data...")
    data = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "username": ["alice", "bob", "charlie", "diana", "eve"],
        "email": [
            "alice@example.com",
            "bob@test.org",
            "invalid-email",  # Invalid
            "diana@company.net",
            "eve@domain.io"
        ],
        "age": [25, 30, 17, 45, 150],  # 17 and 150 are invalid
        "is_active": [True, True, False, True, False]
    })
    print(f"✓ Loaded {len(data)} records")
    print()

    # Step 3: Create validator and validate
    print("Step 3: Validating data...")
    validator = UnifiedValidator(schema.to_validation_schema(), lazy=True)
    result = validator.validate(data)

    print(f"Validation Result: {'✓ VALID' if result.is_valid else '✗ INVALID'}")
    print(f"Total Errors: {len(result.errors)}")
    print(f"Total Warnings: {len(result.warnings)}")
    print()

    # Step 4: Display errors
    if result.errors:
        print("Errors Found:")
        print("-" * 80)
        for i, error in enumerate(result.errors[:5], 1):
            print(f"{i}. Row {error.row}, Column '{error.column}': {error.message[:100]}")
        print()

    # Step 5: Generate reports
    print("Step 5: Generating validation reports...")
    reporter = ValidationReporter(result)

    # Console report
    print("\nConsole Report:")
    print("-" * 80)
    reporter.to_console(verbose=False)

    # Export to files
    reporter.to_json("validation_basic.json")
    reporter.to_html("validation_basic.html")
    print(f"\n✓ Reports exported: validation_basic.json, validation_basic.html")

    # Step 6: Analyze errors with DataFrame
    print("\nStep 6: Analyzing errors...")
    errors_df = reporter.to_dataframe()
    if not errors_df.empty:
        print("\nError Distribution by Column:")
        print(errors_df["column"].value_counts())

    print()
    print("=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
