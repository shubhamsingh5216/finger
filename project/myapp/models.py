# app/models.py
from django.db import models

class Fingerprint(models.Model):
    name = models.CharField(max_length=100)
    image = models.ImageField(upload_to="fingerprints/")
    # You can store descriptors later as binary if you want
    def __str__(self):
        return self.name
