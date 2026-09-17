from celery import Celery
from apps.api.settings import settings

celery = Celery("vln", broker=settings.redis_url, include=["workers.tasks"])
celery.conf.update(task_acks_late=True, task_reject_on_worker_lost=True,
                   worker_prefetch_multiplier=1, task_track_started=True,
                   broker_transport_options={"visibility_timeout": 86400},
                   task_serializer="json", accept_content=["json"], timezone="UTC",
                   beat_schedule={"recover-queued": {"task": "workers.tasks.reconcile", "schedule": 30.0}})
