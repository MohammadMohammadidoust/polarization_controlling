import os
import sys
import logging.config

def configure_logging(log_conf_dict):
    base_path = os.path.dirname(os.path.abspath(__file__))
    try:
        main_script = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    except Exception:
        main_script = "interactive"

    for handler in log_conf_dict.get("handlers", {}).values():
        if "filename" in handler:
            original_filename = handler["filename"]
            base_name, ext = os.path.splitext(original_filename)
            new_filename = f"{base_name}_{main_script}{ext}"
            handler["filename"] = os.path.join(base_path, new_filename)

    logging.config.dictConfig(log_conf_dict)
