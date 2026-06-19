from transformers import TrainerCallback

class TeacherEMACallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Called exactly once per optimizer step, after gradient accumulation
        model.update_teacher()
