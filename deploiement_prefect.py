from prefect import serve

from pipeline_prefect import all_flow, evaluate_flow, train_flow

if __name__ == "__main__":
    serve(
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 8 * * *",
        ),
        train_flow.to_deployment(
            name="ml-pipeline-train",
            cron="0 9 * * *",
        ),
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate",
            cron="0 10 * * *",
        ),
    )
