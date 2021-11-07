import requests
from allauth.socialaccount.models import SocialToken


def get_required_details(user: object) -> list:
    """[summary]

    Args:
        user (object): user instance

    Returns:
        list: [headres,user uuid by the provider]
    """

    social = SocialToken.objects.select_related("account").get(account__user=user)
    return [
        {"content-type": "application/json", "Authorization": f"Bearer {social.token}"},
        social.account.uid,
    ]


def get_messages(user: object, max: int = 100) -> list:
    """[summary]

    Args:
        user (object): user object
        id (str): id if any specifc message required
        max (int): max limit

    Returns:
        list: list of snap shot messages
    """
    try:

        details = get_required_details(user)
        header = details[0]
        url = f"https://gmail.googleapis.com/gmail/v1/users/{details[1]}/threads/?maxResults={max}"
        response = requests.get(url=url, headers=header)
        return response.json()["threads"]

    except Exception as e:
        return []


def get_message(user: object, id: str) -> dict:
    """[summary]

    Args:
        user (object): user object
        id (str): id if any specifc message required

    Returns:
        list: list of snap shot messages
    """
    try:

        details = get_required_details(user)
        header = details[0]
        url = f"https://gmail.googleapis.com/gmail/v1/users/{details[1]}/threads/{id}"
        response = requests.get(url=url, headers=header)
        return response.json()

    except Exception as e:
        return {}
