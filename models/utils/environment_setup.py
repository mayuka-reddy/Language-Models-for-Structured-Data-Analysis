"""
Environment validation utility for ML model training and evaluation.
Checks for required packages, versions, GPU availability, and model loading capabilities.
"""

import sys
import subprocess
import importlib
import platform
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class EnvironmentValidator:
    """
    Comprehensive environment validation for ML performance demo.
    
    Validates dependencies, hardware, and system requirements
    for running the complete ML performance demonstration.
    """
    
    def __init__(self):
        self.validation_results = {}
        self.required_packages = self._get_required_packages()
        self.optional_packages = self._get_optional_packages()
    
    def _get_required_packages(self) -> Dict[str, str]:
        """Define required packages and their minimum versions."""
        return {
            "torch": "1.9.0",
            "transformers": "4.20.0",
            "datasets": "2.0.0",
            "pandas": "1.3.0",
            "numpy": "1.21.0",
            "matplotlib": "3.5.0",
            "seaborn": "0.11.0",
            "sqlparse": "0.4.0",
            "yaml": "6.0",
            "tqdm": "4.60.0"
        }
    
    def _get_optional_packages(self) -> Dict[str, str]:
        """Define optional packages that enhance functionality."""
        return {
            "sentence-transformers": "2.2.0",
            "faiss-cpu": "1.7.0",
            "rank-bm25": "0.2.0",
            "wandb": "0.12.0",
            "plotly": "5.0.0",
            "loguru": "0.6.0",
            "nltk": "3.7",
            "scikit-learn": "1.0.0",
            "jupyter": "1.0.0",
            "ipywidgets": "7.6.0"
        }
    
    def validate_environment(self) -> Dict[str, Any]:
        """
        Run complete environment validation.
        
        Returns:
            Dictionary containing validation results
        """
        logger.info("Starting environment validation...")
        
        results = {
            "system_info": self._check_system_info(),
            "python_version": self._check_python_version(),
            "required_packages": self._check_required_packages(),
            "optional_packages": self._check_optional_packages(),
            "gpu_availability": self._check_gpu_availability(),
            "memory_info": self._check_memory_info(),
            "model_loading": self._test_model_loading(),
            "data_access": self._check_data_access(),
            "overall_status": "unknown"
        }
        
        # Determine overall status
        results["overall_status"] = self._determine_overall_status(results)
        
        self.validation_results = results
        
        logger.info(f"Environment validation complete. Status: {results['overall_status']}")
        
        return results
    
    def _check_system_info(self) -> Dict[str, Any]:
        """Check basic system information."""
        
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
            "architecture": platform.architecture()[0]
        }
    
    def _check_python_version(self) -> Dict[str, Any]:
        """Check Python version compatibility."""
        
        current_version = sys.version_info
        min_version = (3, 8)
        recommended_version = (3, 9)
        
        is_compatible = current_version >= min_version
        is_recommended = current_version >= recommended_version
        
        return {
            "current_version": f"{current_version.major}.{current_version.minor}.{current_version.micro}",
            "minimum_required": f"{min_version[0]}.{min_version[1]}",
            "recommended": f"{recommended_version[0]}.{recommended_version[1]}",
            "is_compatible": is_compatible,
            "is_recommended": is_recommended,
            "status": "pass" if is_compatible else "fail"
        }
    
    def _check_required_packages(self) -> Dict[str, Any]:
        """Check required package availability and versions."""
        
        package_results = {}
        all_available = True
        
        for package_name, min_version in self.required_packages.items():
            result = self._check_package(package_name, min_version, required=True)
            package_results[package_name] = result
            
            if result["status"] != "pass":
                all_available = False
        
        return {
            "packages": package_results,
            "all_available": all_available,
            "status": "pass" if all_available else "fail"
        }
    
    def _check_optional_packages(self) -> Dict[str, Any]:
        """Check optional package availability."""
        
        package_results = {}
        available_count = 0
        
        for package_name, min_version in self.optional_packages.items():
            result = self._check_package(package_name, min_version, required=False)
            package_results[package_name] = result
            
            if result["status"] == "pass":
                available_count += 1
        
        coverage = available_count / len(self.optional_packages)
        
        return {
            "packages": package_results,
            "available_count": available_count,
            "total_count": len(self.optional_packages),
            "coverage": coverage,
            "status": "good" if coverage > 0.7 else "partial" if coverage > 0.3 else "poor"
        }
    
    def _check_package(self, package_name: str, min_version: str, required: bool = True) -> Dict[str, Any]:
        """Check individual package availability and version."""
        
        try:
            # Try to import the package
            module = importlib.import_module(package_name)
            
            # Get version
            version = None
            for attr in ['__version__', 'version', 'VERSION']:
                if hasattr(module, attr):
                    version = getattr(module, attr)
                    break
            
            # Version comparison (simplified)
            version_compatible = True
            if version and min_version:
                try:
                    current_parts = [int(x) for x in version.split('.')]
                    min_parts = [int(x) for x in min_version.split('.')]
                    
                    # Pad shorter version with zeros
                    max_len = max(len(current_parts), len(min_parts))
                    current_parts.extend([0] * (max_len - len(current_parts)))
                    min_parts.extend([0] * (max_len - len(min_parts)))
                    
                    version_compatible = current_parts >= min_parts
                    
                except (ValueError, AttributeError):
                    # If version parsing fails, assume compatible
                    version_compatible = True
            
            status = "pass" if version_compatible else "version_mismatch"
            
            return {
                "available": True,
                "version": version or "unknown",
                "min_version": min_version,
                "version_compatible": version_compatible,
                "status": status
            }
            
        except ImportError:
            return {
                "available": False,
                "version": None,
                "min_version": min_version,
                "version_compatible": False,
                "status": "missing"
            }
    
    def _check_gpu_availability(self) -> Dict[str, Any]:
        """Check GPU availability and CUDA support."""
        
        gpu_info = {
            "cuda_available": False,
            "cuda_version": None,
            "gpu_count": 0,
            "gpu_names": [],
            "gpu_memory": [],
            "status": "cpu_only"
        }
        
        try:
            import torch
            
            gpu_info["cuda_available"] = torch.cuda.is_available()
            
            if gpu_info["cuda_available"]:
                gpu_info["cuda_version"] = torch.version.cuda
                gpu_info["gpu_count"] = torch.cuda.device_count()
                
                for i in range(gpu_info["gpu_count"]):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)  # GB
                    
                    gpu_info["gpu_names"].append(gpu_name)
                    gpu_info["gpu_memory"].append(gpu_memory)
                
                gpu_info["status"] = "gpu_available"
            
        except ImportError:
            gpu_info["status"] = "torch_not_available"
        except Exception as e:
            gpu_info["status"] = f"error: {str(e)}"
        
        return gpu_info
    
    def _check_memory_info(self) -> Dict[str, Any]:
        """Check system memory information."""
        
        memory_info = {
            "total_ram_gb": "unknown",
            "available_ram_gb": "unknown",
            "status": "unknown"
        }
        
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            memory_info["total_ram_gb"] = round(memory.total / (1024**3), 2)
            memory_info["available_ram_gb"] = round(memory.available / (1024**3), 2)
            
            # Determine status based on available memory
            if memory_info["available_ram_gb"] >= 8:
                memory_info["status"] = "sufficient"
            elif memory_info["available_ram_gb"] >= 4:
                memory_info["status"] = "adequate"
            else:
                memory_info["status"] = "limited"
                
        except ImportError:
            memory_info["status"] = "psutil_not_available"
        except Exception as e:
            memory_info["status"] = f"error: {str(e)}"
        
        return memory_info
    
    def _test_model_loading(self) -> Dict[str, Any]:
        """Test model loading capabilities."""
        
        model_tests = {
            "transformers_loading": False,
            "tokenizer_loading": False,
            "model_inference": False,
            "error_messages": [],
            "status": "fail"
        }
        
        try:
            # Test transformers loading
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            model_tests["transformers_loading"] = True
            
            # Test tokenizer loading
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            model_tests["tokenizer_loading"] = True
            
            # Test model loading (lightweight test)
            model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            model_tests["model_inference"] = True
            
            model_tests["status"] = "pass"
            
        except ImportError as e:
            model_tests["error_messages"].append(f"Import error: {str(e)}")
        except Exception as e:
            model_tests["error_messages"].append(f"Model loading error: {str(e)}")
        
        return model_tests
    
    def _check_data_access(self) -> Dict[str, Any]:
        """Check data directory access and sample data availability."""
        
        data_info = {
            "data_dir_exists": False,
            "sample_db_exists": False,
            "writable": False,
            "status": "fail"
        }
        
        try:
            # Check data directory
            data_dir = Path("data")
            data_info["data_dir_exists"] = data_dir.exists()
            
            # Check Olist dataset files
            olist_item_level = data_dir / "olist_item_level.csv"
            olist_order_level = data_dir / "olist_order_level.csv"
            data_info["olist_item_level_exists"] = olist_item_level.exists()
            data_info["olist_order_level_exists"] = olist_order_level.exists()
            data_info["sample_db_exists"] = data_info["olist_item_level_exists"] and data_info["olist_order_level_exists"]
            
            # Check write permissions
            try:
                test_file = data_dir / "test_write.tmp"
                test_file.touch()
                test_file.unlink()
                data_info["writable"] = True
            except:
                data_info["writable"] = False
            
            # Determine status
            if data_info["data_dir_exists"] and data_info["writable"]:
                data_info["status"] = "pass"
            elif data_info["data_dir_exists"]:
                data_info["status"] = "read_only"
            else:
                data_info["status"] = "missing"
                
        except Exception as e:
            data_info["status"] = f"error: {str(e)}"
        
        return data_info
    
    def _determine_overall_status(self, results: Dict[str, Any]) -> str:
        """Determine overall environment status."""
        
        # Check critical requirements
        if results["python_version"]["status"] != "pass":
            return "incompatible"
        
        if results["required_packages"]["status"] != "pass":
            return "missing_dependencies"
        
        if results["model_loading"]["status"] != "pass":
            return "model_loading_failed"
        
        # Check for optimal setup
        gpu_available = results["gpu_availability"]["status"] == "gpu_available"
        memory_sufficient = results["memory_info"]["status"] in ["sufficient", "adequate"]
        optional_coverage = results["optional_packages"]["coverage"] > 0.7
        
        if gpu_available and memory_sufficient and optional_coverage:
            return "optimal"
        elif memory_sufficient and optional_coverage:
            return "good"
        elif results["data_access"]["status"] == "pass":
            return "basic"
        else:
            return "limited"
    
    def generate_setup_report(self) -> str:
        """Generate human-readable setup report."""
        
        if not self.validation_results:
            self.validate_environment()
        
        results = self.validation_results
        
        report_lines = [
            "Environment Validation Report",
            "=" * 50,
            "",
            f"Overall Status: {results['overall_status'].upper()}",
            "",
            "System Information:",
            f"  Platform: {results['system_info']['platform']}",
            f"  Python: {results['python_version']['current_version']} ({results['python_version']['status']})",
            ""
        ]
        
        # GPU Information
        gpu_info = results['gpu_availability']
        if gpu_info['cuda_available']:
            report_lines.extend([
                f"GPU: {gpu_info['gpu_count']} GPU(s) available",
                f"  CUDA Version: {gpu_info['cuda_version']}",
                f"  GPUs: {', '.join(gpu_info['gpu_names'])}"
            ])
        else:
            report_lines.append("GPU: Not available (CPU-only mode)")
        
        report_lines.append("")
        
        # Memory Information
        memory_info = results['memory_info']
        if memory_info['status'] != 'unknown':
            report_lines.append(f"Memory: {memory_info['available_ram_gb']}GB available / {memory_info['total_ram_gb']}GB total")
        
        report_lines.append("")
        
        # Package Status
        req_packages = results['required_packages']
        opt_packages = results['optional_packages']
        
        report_lines.extend([
            "Package Status:",
            f"  Required: {sum(1 for p in req_packages['packages'].values() if p['status'] == 'pass')}/{len(req_packages['packages'])} available",
            f"  Optional: {opt_packages['available_count']}/{opt_packages['total_count']} available ({opt_packages['coverage']:.1%} coverage)"
        ])
        
        # Missing packages
        missing_required = [name for name, info in req_packages['packages'].items() if info['status'] != 'pass']
        if missing_required:
            report_lines.extend([
                "",
                "Missing Required Packages:",
                *[f"  - {pkg}" for pkg in missing_required]
            ])
        
        # Recommendations
        report_lines.extend([
            "",
            "Recommendations:",
            *self._generate_recommendations()
        ])
        
        return "\n".join(report_lines)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate setup recommendations based on validation results."""
        
        recommendations = []
        
        if not self.validation_results:
            return ["Run environment validation first"]
        
        results = self.validation_results
        
        # Python version recommendations
        if results['python_version']['status'] != 'pass':
            recommendations.append("Upgrade Python to version 3.8 or higher")
        elif not results['python_version']['is_recommended']:
            recommendations.append("Consider upgrading to Python 3.9+ for better performance")
        
        # Package recommendations
        missing_required = [
            name for name, info in results['required_packages']['packages'].items() 
            if info['status'] != 'pass'
        ]
        
        if missing_required:
            recommendations.append(f"Install missing packages: pip install {' '.join(missing_required)}")
        
        # GPU recommendations
        if results['gpu_availability']['status'] == 'cpu_only':
            recommendations.append("Consider using GPU for faster training (install CUDA-enabled PyTorch)")
        
        # Memory recommendations
        memory_status = results['memory_info']['status']
        if memory_status == 'limited':
            recommendations.append("Increase available RAM or use smaller batch sizes")
        elif memory_status == 'adequate':
            recommendations.append("Consider increasing RAM for better performance with larger models")
        
        # Optional package recommendations
        if results['optional_packages']['coverage'] < 0.5:
            missing_optional = [
                name for name, info in results['optional_packages']['packages'].items()
                if info['status'] != 'pass'
            ]
            recommendations.append(f"Install optional packages for enhanced functionality: {', '.join(missing_optional[:3])}")
        
        return recommendations
    
    def save_validation_report(self, output_path: str = "models/utils/environment_report.json") -> str:
        """Save validation results to file."""
        
        if not self.validation_results:
            self.validate_environment()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        logger.info(f"Environment validation report saved to {output_path}")
        
        return str(output_file)


# Example usage and testing
if __name__ == "__main__":
    # Test environment validation
    validator = EnvironmentValidator()
    
    print("Running Environment Validation:")
    print("=" * 50)
    
    # Run validation
    results = validator.validate_environment()
    
    # Generate and print report
    report = validator.generate_setup_report()
    print(report)
    
    # Save results
    report_path = validator.save_validation_report()
    print(f"\nDetailed results saved to: {report_path}")
    
    print(f"\nOverall Status: {results['overall_status'].upper()}")