class CommonHelper:
   def __init__(self):
      pass
   @staticmethod
   def readQss(style):
       with open(style, 'r',encoding='utf-8') as f:
          return f.read()