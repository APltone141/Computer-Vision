import json
import os
import threading
from typing import Any, Dict, Optional, List
from pathlib import Path
import logging
import time

class ConfigManager:
    """
    Configuration manager untuk CameraAI application
    Mengelola settings, user preferences, dan module configurations
    """
    
    def __init__(self, config_file: str = "config.json", auto_save: bool = True):
        self.config_file = config_file
        self.auto_save = auto_save
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Default configuration
        self._default_config = {
            # Camera settings
            'camera': {
                'default_index': 0,
                'width': 640,
                'height': 480,
                'fps': 30,
                'queue_size': 64
            },
            
            # Module settings
            'modules': {
                'autofocus': {
                    'enabled': False,
                    'min_confidence': 0.6,
                    'padding': 50,
                    'alpha': 0.4,
                    'min_crop_ratio': 0.6
                },
                'tracking': {
                    'enabled': False,
                    'mode': 'face',
                    'min_confidence': 0.6,
                    'smoothing_factor': 0.3,
                    'zoom_factor': 1.0,
                    'max_offset': 0.3
                },
                'motion_detection': {
                    'enabled': False,
                    'sensitivity': 50,
                    'min_area': 500,
                    'history_length': 10
                },
                'gesture_control': {
                    'enabled': False,
                    'min_detection_confidence': 0.7,
                    'min_tracking_confidence': 0.5,
                    'cooldown': 1.0
                },
                'super_resolution': {
                    'enabled': False,
                    'scale_factor': 2.0,
                    'model_name': 'RealESRGAN_x4plus'
                },
                'fps_interpolator': {
                    'enabled': False,
                    'target_fps': 60.0,
                    'interpolation_method': 'opencv'
                }
            },
            
            # GUI settings
            'gui': {
                'window_width': 1200,
                'window_height': 800,
                'theme': 'default',
                'show_fps': True,
                'show_stats': True,
                'auto_start_camera': True
            },
            
            # Performance settings
            'performance': {
                'max_fps': 60,
                'quality_mode': 'balanced',  # balanced, quality, speed
                'enable_logging': True,
                'log_level': 'INFO',
                'enable_performance_monitoring': True
            },
            
            # Recording settings
            'recording': {
                'output_dir': 'recordings',
                'format': 'mp4',
                'codec': 'H264',
                'quality': 'high',
                'auto_record': False
            },
            
            # AI models
            'ai_models': {
                'face_detection': 'mediapipe',  # mediapipe, opencv
                'hand_detection': 'mediapipe',
                'super_resolution': 'realesrgan',  # realesrgan, opencv
                'fps_interpolation': 'opencv'  # opencv, linear, cubic, advanced
            },
            
            # Advanced settings
            'advanced': {
                'enable_parallel_processing': True,
                'max_workers': 3,
                'enable_fallback': True,
                'debug_mode': False,
                'save_screenshots': False,
                'screenshot_dir': 'screenshots'
            }
        }
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                # Merge with default config (preserve new defaults)
                merged_config = self._merge_configs(self._default_config, loaded_config)
                self.logger.info(f"Configuration loaded from {self.config_file}")
                return merged_config
            else:
                # Create default config file
                self.config = self._default_config.copy()
                self._save_config()
                self.logger.info(f"Default configuration created: {self.config_file}")
                return self.config
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return self._default_config.copy()
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Merge loaded config with default config"""
        merged = default.copy()
        
        def merge_dict(base: Dict[str, Any], update: Dict[str, Any]):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(merged, loaded)
        return merged
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            self.logger.debug("Configuration saved")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        try:
            keys = key.split('.')
            value = self.config
            
            for k in keys:
                value = value[k]
            
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any, auto_save: bool = None):
        """Set configuration value using dot notation"""
        try:
            keys = key.split('.')
            config = self.config
            
            # Navigate to parent of target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set value
            config[keys[-1]] = value
            
            # Auto save if enabled
            if auto_save is None:
                auto_save = self.auto_save
            
            if auto_save:
                self._save_config()
                
        except Exception as e:
            self.logger.error(f"Failed to set configuration {key}: {e}")
    
    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """Get configuration for specific module"""
        return self.get(f'modules.{module_name}', {})
    
    def set_module_config(self, module_name: str, config: Dict[str, Any]):
        """Set configuration for specific module"""
        self.set(f'modules.{module_name}', config)
    
    def get_camera_config(self) -> Dict[str, Any]:
        """Get camera configuration"""
        return self.get('camera', {})
    
    def set_camera_config(self, config: Dict[str, Any]):
        """Set camera configuration"""
        self.set('camera', config)
    
    def get_gui_config(self) -> Dict[str, Any]:
        """Get GUI configuration"""
        return self.get('gui', {})
    
    def set_gui_config(self, config: Dict[str, Any]):
        """Set GUI configuration"""
        self.set('gui', config)
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.get('performance', {})
    
    def set_performance_config(self, config: Dict[str, Any]):
        """Set performance configuration"""
        self.set('performance', config)
    
    def get_recording_config(self) -> Dict[str, Any]:
        """Get recording configuration"""
        return self.get('recording', {})
    
    def set_recording_config(self, config: Dict[str, Any]):
        """Set recording configuration"""
        self.set('recording', config)
    
    def get_ai_models_config(self) -> Dict[str, Any]:
        """Get AI models configuration"""
        return self.get('ai_models', {})
    
    def set_ai_models_config(self, config: Dict[str, Any]):
        """Set AI models configuration"""
        self.set('ai_models', config)
    
    def get_advanced_config(self) -> Dict[str, Any]:
        """Get advanced configuration"""
        return self.get('advanced', {})
    
    def set_advanced_config(self, config: Dict[str, Any]):
        """Set advanced configuration"""
        self.set('advanced', config)
    
    def reset_to_defaults(self, section: Optional[str] = None):
        """Reset configuration to defaults"""
        if section:
            if section in self._default_config:
                self.config[section] = self._default_config[section].copy()
                self.logger.info(f"Reset {section} to defaults")
        else:
            self.config = self._default_config.copy()
            self.logger.info("Reset all configuration to defaults")
        
        self._save_config()
    
    def export_config(self, output_file: str):
        """Export configuration to file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Configuration exported to {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
    
    def import_config(self, input_file: str):
        """Import configuration from file"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
            
            # Merge with current config
            self.config = self._merge_configs(self.config, imported_config)
            self._save_config()
            self.logger.info(f"Configuration imported from {input_file}")
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate camera settings
        camera_config = self.get_camera_config()
        if camera_config.get('width', 0) <= 0:
            issues.append("Camera width must be positive")
        if camera_config.get('height', 0) <= 0:
            issues.append("Camera height must be positive")
        if camera_config.get('fps', 0) <= 0:
            issues.append("Camera FPS must be positive")
        
        # Validate module settings
        for module_name, module_config in self.get('modules', {}).items():
            if isinstance(module_config, dict):
                for key, value in module_config.items():
                    if key == 'enabled' and not isinstance(value, bool):
                        issues.append(f"{module_name}.enabled must be boolean")
                    elif key in ['min_confidence', 'alpha', 'smoothing_factor'] and not (0 <= value <= 1):
                        issues.append(f"{module_name}.{key} must be between 0 and 1")
        
        # Validate performance settings
        perf_config = self.get_performance_config()
        if perf_config.get('max_fps', 0) <= 0:
            issues.append("Max FPS must be positive")
        if perf_config.get('quality_mode') not in ['balanced', 'quality', 'speed']:
            issues.append("Quality mode must be balanced, quality, or speed")
        
        return issues
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all configuration settings"""
        return self.config.copy()
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        self.config = self._merge_configs(self.config, config_dict)
        self._save_config()
        self.logger.info("Configuration updated from dictionary")
    
    def create_backup(self, backup_file: str = None):
        """Create configuration backup"""
        if backup_file is None:
            timestamp = int(time.time())
            backup_file = f"config_backup_{timestamp}.json"
        
        self.export_config(backup_file)
        return backup_file
    
    def restore_backup(self, backup_file: str):
        """Restore configuration from backup"""
        self.import_config(backup_file)
    
    def __del__(self):
        """Cleanup on deletion"""
        if self.auto_save:
            self._save_config()


# Global config manager instance
_config_instance = None

def get_config() -> ConfigManager:
    """Get global config manager instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance

def setup_config(config_file: str = "config.json", **kwargs) -> ConfigManager:
    """Setup global config manager with custom parameters"""
    global _config_instance
    _config_instance = ConfigManager(config_file, **kwargs)
    return _config_instance 