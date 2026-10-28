"""A deliberately named smoke-test policy for the 2D demo adapter only."""
import asyncio
import numpy as np
from rilo.contracts import ActionCommand, Decision


class ScriptedPolicy:
    async def decide(self, observation, spec, context, history):
        await asyncio.sleep(0.5)
        values = np.clip((np.array(observation.state["target"]) - observation.state["position"]) / 0.1, -1, 1)
        return Decision(observation_id=observation.id, status="continue",
                        actions=[ActionCommand(values=values.tolist(), repeat=3)],
                        note="Scripted demo: moving toward the target. No GPT request was made.")

    async def close(self):
        pass
