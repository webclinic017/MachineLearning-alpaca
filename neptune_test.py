import neptune.new as neptune

run = neptune.init_run(
    project = "common/quickstarts",
    api_token = "ANONYMOUS"
)

# Track metadata and hyperparameters of your run
run["JIRA"] = "NPT-952"
run["algorithm"] = "ConvNet"

params = {
    "batch_size": 64,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "optimizer": "Adam",
}
run["parameters"] = params

for epoch in range(100):
    run["train/accuracy"].log(epoch * 0.6)
    run["train/loss"].log(epoch * 0.4)
    # this just creates a couple of series so we can view them as charts

# Log the final results
run["f1_score"] = 0.66

# Stop the connection and sync the data with Neptune
run.stop()
