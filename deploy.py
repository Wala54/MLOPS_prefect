"""
deploy.py
---------
Expose le flow complet (flow_all) comme un deployment Prefect,
avec un planning quotidien (comme dans la slide 13).

Lancer avec :
    python deploy.py

Cela démarre un worker local et active le scheduler : le flow
s'exécutera automatiquement à l'heure planifiée, tant que ce
processus tourne (ou via `prefect deployment run` pour un
déclenchement manuel).
"""

from prefect.client.schemas.schedules import CronSchedule

from pipeline import flow_all

if __name__ == "__main__":
    flow_all.serve(
        name="ml-pipeline-all",
        schedules=[
            CronSchedule(cron="0 2 * * *", timezone="Africa/Tunis")
        ],
        tags=["mlops", "full-pipeline"],
    )
