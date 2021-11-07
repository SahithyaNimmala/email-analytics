import uuid
from django.db import models
from django.db.models.fields import CharField, EmailField, UUIDField
from django.contrib.auth.models import AbstractUser,BaseUserManager
from django.utils.translation import ugettext_lazy as _


class UserManager(BaseUserManager):
    """Helps in creating a user
    Args:
        BaseUserManager ([type]): [description]
    Raises:
        ValueError: if the user status is not matched
    Returns:
        User: User object in django
    """

    use_in_migrations = True

    def _create_user(self, email: str, password: str, **extra_fields):
        """Create and save a User with  given email and password.
        Raises:
            ValueError: if email is null
        Returns:
            User: User object in django
        """
        if not email:
            raise ValueError("Please provide a valid email")
        user = self.model(email=self.normalize_email(email), **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_user(self, email: str, password: str = None, **extra_fields):
        """Create & save a normal User with  given email and password
        Returns:
            User: User object in django
        """
        extra_fields.setdefault("is_staff", False)
        extra_fields.setdefault("is_superuser", False)
        return self._create_user(email, password, **extra_fields)

    def create_superuser(self, email: str, password: str, **extra_fields):
        """This will create super user 
        Args:
            email str: email 
            password str: password
            
        Raises:
            ValueError: email is not correct 
            
        Returns:
            User: User object in django
        """
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)

        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser must have is_staff True.")
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser True.")

        return self._create_user(email, password, **extra_fields)

class User(AbstractUser):

    username = None
    id = UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = EmailField(_("email"), max_length=254, unique=True)
    image = models.ImageField(upload_to="profile_picture", null=False, blank=False)
    force_password_change = models.BooleanField(default=False)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []

    objects = UserManager()
    
    def __str__(self):
        return self.email

    class Meta:
        ordering = ["id"]
