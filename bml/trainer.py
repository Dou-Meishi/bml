import tqdm

from .utils import add_column_to_record


class Trainer(object):
    
    def __init__(self, steps_per_epoch, max_epoches=1, *,
                 lr_decay_per_epoch=1.0, warm_up_lr=None):
        self.steps_per_epoch = steps_per_epoch
        self.max_epoches = max_epoches
        self.lr_decay_per_epoch = lr_decay_per_epoch
        self.warm_up_lr = warm_up_lr
        
    def create_lr_schedule(self, usual_lr):
        # Initialize the learning rate schedule list with warm-up learning rates
        lr_schedule = self.warm_up_lr if self.warm_up_lr is not None else []

        # Calculate the number of remaining epochs after the warm-up period
        remaining_epochs = self.max_epoches - len(self.warm_up_lr)

        # Generate the decaying learning rate schedule for the remaining epochs
        for epoch in range(remaining_epochs):
            decayed_lr = usual_lr * (1 - self.lr_decay_per_epoch) ** epoch
            lr_schedule.append(decayed_lr)

        return lr_schedule

    def train(self, solver):
        # create progress bar
        pbar = tqdm.tqdm(total=self.steps_per_epoch,
                         desc=f"Epoch: 1, Val Loss: 0.0")

        tab_logs = []    # summary metrics at the end of the training
        fig_logs = []    # running metrics during training

        lr_schedule = self.create_lr_schedule(solver.lr)

        for epoch in range(self.max_epoches):
            if epoch > 0:  # reset the progress bar at the start of each epoch
                pbar.reset(total=self.steps_per_epoch)
                pbar.set_description(
                    f"Epoch: {epoch + 1}/{self.max_epoches}, Val Loss: {val_loss:.4f}")

            # select lr
            solver.lr = lr_schedule[epoch]

            tab_log, fig_log = solver.solve(self.steps_per_epoch, pbar)

            val_loss = tab_log[-1]['val loss']

            # add a column to record the current number of epoches
            tab_logs += add_column_to_record(tab_log, 'epoch', [epoch] * len(tab_log))
            fig_logs += add_column_to_record(fig_log, 'epoch', [epoch] * len(fig_log))

        # update the final val loss and close
        pbar.set_description(f"Epoch: {epoch + 1}/{self.max_epoches}, Val Loss: {val_loss:.4f}")
        pbar.close()
        
        return tab_logs, fig_logs

