from django.db import models

class Stock(models.Model):
    Apple = 'AAPL'
    Google = 'GOOGL'
    Microsoft = 'MSFT'

    STOCK_CHOICES = [
        (Apple, 'AAPL'),
        (Google, 'GOOGL'),
        (Microsoft, 'MSFT'),
    ]

    name = models.CharField(
        max_length=10, choices=STOCK_CHOICES, default=Google)
