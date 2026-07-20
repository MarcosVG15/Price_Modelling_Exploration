from abc import ABC , abstractmethod

class Agent(ABC) :
    
    @abstractmethod
    def observe(self,  s, a, r, s2, done):
        pass

    @abstractmethod
    def choose(self, state, explore=True):
        """Return an action. explore=True → exploring (ε-greedy / sample);
           explore=False → greedy (argmax / policy mean). Used by both train and eval."""

    def on_episode_end(self):
        """Optional hook called at each episode boundary: anneal exploration or run
        episodic updates. No-op by default so agents that don't need it can skip it."""
        pass

