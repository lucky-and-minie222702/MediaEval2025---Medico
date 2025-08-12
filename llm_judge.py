from my_tools import *

config = MyConfig.load_json(sys.argv[1])
checkpoint = config.get("checkpoint", MyUtils.get_latest_checkpoint(config['dir']))
model_name = config.get("model_name", "cross-encoder/stsb-roberta-large")
agent = MyUtils.TestLogger.LLMJudgeAgent(
    dir = config["dir"],
    checkpoint = checkpoint,
    model_name = model_name,
)
agent.calc_scores()
agent.from_csv_to_csv(
    file1 = "data/test.csv",
    file2 = f"results/{config["dir"]}/checkpoint-{checkpoint}-test-stats.csv"
)