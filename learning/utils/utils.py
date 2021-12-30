def print_and_log(msg: str, logger):
    if logger is not None:
        print(msg)
        logger.info(msg)
    else:
        print(msg)