#!/usr/bin/env python3
"""
Environment validation script for ML Model Performance Demo.
Run this script to check if your environment is properly set up.
"""

import sys
from pathlib import Path

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.environment_setup import EnvironmentValidator


def main():
    """Main validation function."""
    print("ML Model Performance Demo - Environment Validation")
    print("=" * 60)
    print()
    
    # Create validator and run validation
    validator = EnvironmentValidator()
    results = validator.validate_environment()
    
    # Print report
    report = validator.generate_setup_report()
    print(report)
    
    # Save detailed results
    report_path = validator.save_validation_report()
    print(f"\nDetailed validation results saved to: {report_path}")
    
    # Exit with appropriate code
    status = results['overall_status']
    if status in ['optimal', 'good', 'basic']:
        print(f"\n✅ Environment validation PASSED ({status})")
        sys.exit(0)
    else:
        print(f"\n❌ Environment validation FAILED ({status})")
        print("\nPlease address the issues above before running the ML performance demo.")
        sys.exit(1)


if __name__ == "__main__":
    main()