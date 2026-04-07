from app.models import Observation, Action
from app.data.generator import generate_user_data
from app.tasks.graders import grade_action


class FinAgentEnv:
    def __init__(self):
        self.state_data = None
        self.steps = 0
        self.max_steps = 20

    def reset(self):
        self.state_data = generate_user_data()
        self.steps = 0
        observation = Observation(**self.state_data)
        return observation.dict()

    def state(self):
        return self.state_data

    def step(self, action_dict):
        try:
            action = Action(**action_dict)
        except Exception as e:
            return {
                "observation": self.state_data,
                "reward": 0.0,
                "done": False,
                "info": {"error": str(e)}
            }

        reward, info = grade_action(self.state_data, action)
        reward = float(max(0.0, min(1.0, reward)))

        self.steps += 1
        done = self.steps >= self.max_steps

        observation = Observation(**self.state_data)

        return {
            "observation": observation.dict(),
            "reward": reward,
            "done": bool(done),
            "info": info if isinstance(info, dict) else {}
        }
