import logging.config

def configure_logging(log_conf_dict):
    logging.config.dictConfig(log_conf_dict)
