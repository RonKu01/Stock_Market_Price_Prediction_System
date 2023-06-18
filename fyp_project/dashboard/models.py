from django.db import models

class Stock(models.Model):
    Apple = 'AAPL'
    Google = 'GOOGL'
    Microsoft = 'MSFT'

    STOCK_CHOICES = [
        (Google, 'GOOGL'),
        (Apple, 'AAPL'),
        (Microsoft, 'MSFT'),
    ]


    userType = models.CharField(max_length=10)
    
    name = models.CharField(
        max_length=10, choices=STOCK_CHOICES, default=Google)
    
    
