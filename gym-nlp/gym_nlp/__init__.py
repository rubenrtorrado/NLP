import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='NLP-GYM',
    entry_point='gym_nlp.envs:NLPEnv',
)
