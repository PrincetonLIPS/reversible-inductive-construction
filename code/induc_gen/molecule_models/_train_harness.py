import pickle
import time
import torch

from .. import model as mo, molecule_representation as mr
from . import _train_utils


def _log_if_rank_zero(msg):
    from torch import distributed as dist
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return

    print(msg)


def _aggregate_distributed_batchlike(values):
    if not _train_utils.is_distributed():
        return values, None

    from torch import distributed as dist

    world_size = dist.get_world_size()
    all_values = values.new_empty([values.shape[0] * world_size] + list(values.shape[1:]))
    async_handle = dist.all_gather(
        list(torch.chunk(all_values, world_size)), values, async_op=True)
    return all_values, async_handle


def compute_and_aggregate_predictions(logits_and_scopes, batch, keys=None):
    if keys is None:
        keys = ['label', 'label_offsets', 'action_vector']

    with torch.no_grad():
        predictions, _ = mo.classification.multi_classification_prediction(
            logits_and_scopes, True)
        predictions = predictions.int()

    predictions, a0 = _aggregate_distributed_batchlike(predictions)

    result = [predictions]
    async_handles = [a0]

    for k in keys:
        r, a = _aggregate_distributed_batchlike(batch[k])
        result.append(r)
        async_handles.append(a)

    for async_handle in async_handles:
        if async_handle is not None:
            async_handle.wait()

    return result


class TrainingHarness:
    def __init__(self, model, optimizer, summary, task='train', profile=False):
        self.model = model
        self.optimizer = optimizer
        self.summary = summary
        self.hooks = []
        self.global_step = 0
        self.epoch_progress = 0.0
        self.epoch = 0
        self.task = task
        self.profile = profile

    def get_model_input(self, batch):
        graph = _train_utils.replace_sparse_tensor(mr.GraphInfo.from_sequence(batch['graph']))
        return graph,

    def get_loss(self, model_output, batch):
        return mo.classification.multi_classification_loss(model_output, batch['label'])

    def get_logits_and_scopes(self, model_output):
        return model_output

    def record_step_summary(self, batch, model_output):
        logits_and_scopes = self.get_logits_and_scopes(model_output)
        prediction_and_info = compute_and_aggregate_predictions(logits_and_scopes, batch)

        if self.summary:
            self.summary.record_statistics(*prediction_and_info)

    def step_single(self, batch, predict=False, profile=None):
        if self.task == 'train':
            self.optimizer.zero_grad()

        batch = _train_utils.load_cuda_async(batch)

        with torch.autograd.profiler.profile(enabled=bool(profile), use_cuda=True) as trace:
            with torch.autograd.set_grad_enabled(self.task == 'train'):
                model_input = self.get_model_input(batch)
                model_output = self.model(*model_input)
                loss = self.get_loss(model_output, batch)

            if self.task == 'train':
                with mo.autograd_range("backward"):
                    loss.backward()
                with mo.autograd_range("optimizer_update"):
                    self.optimizer.step()

        if profile is not None:
            with open(profile, 'wb') as f:
                pickle.dump(trace, f, pickle.HIGHEST_PROTOCOL)

        if predict:
            self.record_step_summary(batch, model_output)

        self.global_step += 1

        return loss

    def train_epoch(self, dataloader):
        for it, batch in enumerate(dataloader):
            self.epoch_progress = it / len(dataloader)
            predict = (self.task == 'test') or ((it % 10) == 0)

            if self.profile and (it % 15) == 0:
                profile_path = 'trace_{0}'.format(it)
            else:
                profile_path = None

            loss = self.step_single(batch, predict, profile_path)
            self._invoke_hooks({'epoch': self.epoch, 'it': it, 'loss': loss, 'global_step': self.global_step})

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _invoke_hooks(self, info):
        for hook in self.hooks:
            hook(info)


class LogLossTimeHook:
    def __init__(self, batch_size, log_freq=10, summary_writer=None):
        self.batch_size = batch_size
        self.log_freq = log_freq
        self.summary_writer = summary_writer
        self._last_tic_time = time.perf_counter()
        self._last_tic_iter = 0

    def __call__(self, info):
        if info['it'] % self.log_freq != 0:
            return

        loss = info['loss']
        global_step = info['global_step']

        current_time = time.perf_counter()
        elapsed_time = current_time - self._last_tic_time
        elapsed_iter = global_step - self._last_tic_iter

        mean_time = elapsed_time / elapsed_iter / self.batch_size * 1e3
        _log_if_rank_zero(
            "Mean time: {0:.2f} ms / mol, Epoch: {1}, It: {2}, Loss: {3:.3f}".format(
                mean_time, info['epoch'], info['it'], loss))

        if self.summary_writer is not None:
            self.summary_writer.add_scalar('loss', loss, global_step)
            self.summary_writer.add_scalar('ms_per_mol', mean_time, global_step)

            self._last_tic_time = current_time
            self._last_tic_iter = global_step


class PrintAccuracyHook:
    def __init__(self, summary, summary_writer, task='train'):
        self.summary = summary
        self.summary_writer = summary_writer
        self.task = task

    def __call__(self, info):
        if (info['it'] + 1) % 50 != 0:
            return

        if self.summary is not None:
            print(self.summary.get_string_representation())

            if self.summary_writer is not None:
                self.summary.write_tensorboard(self.summary_writer, "metrics", global_step=info['global_step'])

        if self.task == 'train' and self.summary is not None:
            self.summary.reset_statistics()


class LogModelWeightsHook:
    def __init__(self, model, summary_writer=None):
        self.model = model
        self.summary_writer = summary_writer

    def __call__(self, info):
        if info['it'] % 50 != 0:
            return

        if self.summary_writer is not None:
            model_to_tensorboard_histogram(
                self.model, self.summary_writer, 'model', global_step=info['global_step'])


class LogOptimizerHook:
    def __init__(self, optimizer, summary_writer=None):
        self.optimizer = optimizer
        self.summary_writer = summary_writer

    def __call__(self, info):
        if info['it'] % 10 != 0 or self.summary_writer is None:
            return

        self.summary_writer.add_scalar(
            'learning_rate', self.optimizer.param_groups[0]['lr'], global_step=info['global_step'])


def model_to_tensorboard_histogram(model, summary_writer, prefix="", **kwargs):
    for name, param in model.named_parameters(prefix):
        summary_writer.add_histogram(name.replace('.', '/'), param, bins='tensorflow', **kwargs)
