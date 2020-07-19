from Env.SMARTS_env import SMARTS_env
from RL_Agent.RL_Agent_Handler import RL_Agent_Handler
from Env.Wrapper_env import  Wrapper
import  time

env = SMARTS_env()
wrapper = Wrapper(env)
wrapper.partition_env()


handler = RL_Agent_Handler(wrapper)
handler.create_agents(pre_trained=True)
#handler.load_weights()
start = time.time()
while True:
    handler.predict(120)
    end = time.time()
    print("Exec time", end - start)
    wrapper.partition_env()
#handler.learn(60)


