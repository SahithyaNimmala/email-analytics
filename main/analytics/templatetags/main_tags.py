from django import template
import base64


register = template.Library()



@register.filter(name="base64")
def mail_decode(data: str) -> list:
    data  =  base64.urlsafe_b64decode(data).decode('utf-8')
    print(data)
    return data