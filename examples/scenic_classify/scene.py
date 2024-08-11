from dotenv import load_dotenv
from autolabel import LabelingAgent, AutolabelDataset, get_data
import json
import os
load_dotenv()  # 默认从项目根目录下的.env文件加载
api_key=os.getenv("OPENAI_API_KEY")
url=os.getenv("OPENAI_BASE_URL")
print(url)

config='examples/scenic_classify/config_scene.json'
agent = LabelingAgent(config=config)

ds = AutolabelDataset('examples/scenic_classify/test.csv', config = config)
agent.plan(ds)

agent = LabelingAgent(config)
ds = AutolabelDataset('data/banking77.csv', config = config)
agent.plan(ds)
agent.run(ds, max_items = 100)