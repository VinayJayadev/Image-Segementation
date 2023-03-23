import os
import logging
import datetime
from logging.handlers import RotatingFileHandler

class Logger:

  def __init__(self):
    self.logger = None

  def get_logger(self, log_level, name, extras = {}):
    """
    Create logs based on log file name.
    """
    try:
      self.logger = logging.getLogger(name)
      if len(self.logger.handlers) == 0: 
        logPath = name + '.log'
        hdlr = RotatingFileHandler(filename = logPath , mode='a', maxBytes = 5*1024*1024)
        if extras == {}:
          extras["appName"] = "clustering"
          formatter = logging.Formatter(fmt='%(appName)s - %(asctime)s [%(threadName)s] %(levelname)s %(filename)s %(funcName)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        else:
          formatter = logging.Formatter(fmt='%(appName)s - %(asctime)s [%(threadName)s] %(levelname)s %(filename)s %(funcName)s - [%(tenantId)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        hdlr.setFormatter(formatter)        
        extras["appName"] = "clustering"
        self.logger.addHandler(hdlr)
        levels = (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL)
        for each_level in levels:
          if logging.getLevelName(each_level) == log_level:
            self.logger.setLevel(each_level)
      self.logger = logging.LoggerAdapter(self.logger,extras)
      return self.logger
    except Exception as LoggerCreationError:
      raise LoggerCreationError