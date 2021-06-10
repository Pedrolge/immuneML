import abc


class Output(metaclass=abc.ABCMeta):

    def set_context(self, context: dict):
        return self
