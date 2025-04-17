import torch
import utils
import pdb
# from line_profiler import LineProfiler


class BeamSearch(object):
    def __init__(self, model, max_len: int, eos_idx: int, beam_size: int):
        self.model = model
        self.max_len = max_len
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.b_s = None
        self.device = None
        self.seq_mask = None
        self.seq_logprob = None
        self.outputs = None
        self.log_probs = None
        self.selected_words = None
        self.all_log_probs = None

    def _expand_state(self, selected_beam, cur_beam_size):
        def fn(s):
            # print('-' * 50)
            # print('input size:', s.shape)
            tensor_in = s.clone()
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            # print(beam)
            # print(beam.dtype)
            # print(beam.shape)

            tmp_s_view = s.view(*([self.b_s, cur_beam_size] + shape[1:]))
            tmp_beam_expand = beam.expand(*([self.b_s, self.beam_size] + shape[1:]))
            s = torch.gather(tmp_s_view, 1, tmp_beam_expand)
            final_s = s.view(*([-1, ] + shape[1:]))
            # print('output size:', final_s.shape)
            if tensor_in.shape == final_s.shape:
                print(torch.sum(torch.abs(tensor_in.float() - final_s.float())) / torch.numel(final_s))
            pdb.set_trace()
            return final_s

        return fn

    def edit_running_caches(self, modules, cache_names, selected_beam, cur_beam_size):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)

            tmp_s_view = s.view(*([self.b_s, cur_beam_size] + shape[1:]))
            tmp_beam_expand = beam.expand(*([self.b_s, self.beam_size] + shape[1:]))
            s = torch.gather(tmp_s_view, 1, tmp_beam_expand)
            final_s = s.view(*([-1, ] + shape[1:]))
            return final_s

        for module in modules:
            for cache_name in cache_names:
                module._buffers[cache_name] = fn(module._buffers[cache_name])

    def _simple_expand(self):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            target_shape = [self.b_s, self.beam_size, *shape[1:]]
            s = s.unsqueeze(1)
            s = s.expand(target_shape)
            s = s.contiguous().flatten(0, 1)
            return s

        return fn

    def _expand_visual(self, visual: utils.TensorOrSequence, cur_beam_size: int, selected_beam: torch.Tensor):
        if isinstance(visual, torch.Tensor):
            visual_shape = visual.shape
            visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
            visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
            selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2))
            selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
            visual_exp = visual.view(visual_exp_shape)
            selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
            visual = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)
        else:
            new_visual = []
            for im in visual:
                visual_shape = im.shape
                visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
                visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
                selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(visual_exp_shape) - 2))
                selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
                visual_exp = im.view(visual_exp_shape)
                selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
                new_im = torch.gather(visual_exp, 1, selected_beam_exp).view(visual_red_shape)
                new_visual.append(new_im)
            visual = tuple(new_visual)
        return visual

    def apply(self, visual: utils.TensorOrSequence, out_size=1, return_probs=False, **kwargs):
        self.b_s = utils.get_batch_size(visual)
        self.device = utils.get_device(visual)
        self.seq_mask = torch.ones((self.b_s, self.beam_size, 1), device=self.device)
        self.seq_logprob = torch.zeros((self.b_s, 1, 1), device=self.device)
        self.log_probs = None
        self.selected_words = None
        if return_probs:
            self.all_log_probs = []

        outputs = None

        with self.model.statefulness(self.b_s):
            for t in range(self.max_len):
                # lp = LineProfiler()
                # target_func = lp(self.iter)
                # visual, outputs = target_func(t, visual, outputs, return_probs, **kwargs)
                visual, outputs = self.iter(t, visual, outputs, return_probs, **kwargs)
                # lp.print_stats()

        # Sort result
        seq_logprob, sort_idxs = torch.sort(self.seq_logprob, 1, descending=True)
        # outputs = torch.cat(outputs, -1) # iter里已经cat起来了
        outputs = torch.gather(outputs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        # log_probs = torch.cat(self.log_probs, -1) # iter里已经cat起来了
        log_probs = self.log_probs
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        if return_probs:
            all_log_probs = torch.cat(self.all_log_probs, 2)
            all_log_probs = torch.gather(all_log_probs, 1, sort_idxs.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                                          self.max_len,
                                                                                          all_log_probs.shape[-1]))

        outputs = outputs.contiguous()[:, :out_size]
        log_probs = log_probs.contiguous()[:, :out_size]
        if out_size == 1:
            outputs = outputs.squeeze(1)
            log_probs = log_probs.squeeze(1)

        if return_probs:
            return outputs, log_probs, all_log_probs
        else:
            return outputs, log_probs

    def select(self, t, candidate_logprob, **kwargs):

        selected_logprob, selected_idx = torch.topk(candidate_logprob.view(self.b_s, -1), self.beam_size, -1)

        return selected_idx, selected_logprob

    def iter(self, t: int, visual: utils.TensorOrSequence, outputs, return_probs, **kwargs):
        cur_beam_size = 1 if t == 0 else self.beam_size

        word_logprob = self.model.step(t, self.selected_words, visual, None, mode='feedback', **kwargs)
        word_logprob = word_logprob.view(self.b_s, cur_beam_size, -1)
        candidate_logprob = self.seq_logprob + word_logprob

        # Mask sequence if it reaches EOS
        if t > 0:
            mask = (self.selected_words.view(self.b_s, cur_beam_size) != self.eos_idx).float().unsqueeze(-1)
            self.seq_mask = self.seq_mask * mask
            word_logprob = word_logprob * self.seq_mask.expand_as(word_logprob)
            old_seq_logprob = self.seq_logprob.expand_as(candidate_logprob).contiguous()
            old_seq_logprob[:, :, 1:] = -999
            candidate_logprob = self.seq_mask * candidate_logprob + old_seq_logprob * (1 - self.seq_mask)

        selected_idx, selected_logprob = self.select(t, candidate_logprob, **kwargs)
        selected_beam = (selected_idx / candidate_logprob.shape[-1]).long()  # ?

        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

        if t == 0:
            self.model.apply_to_states(self._simple_expand())
        else:
            cache_names = ['running_keys', 'running_values']
            target_modules = []
            modules = self.model.decoder.layers
            for module in modules:
                mha_module = module.self_att
                target_modules.append(mha_module)
            # print(len(target_modules))
            # print(print(target_modules))
            self.edit_running_caches(target_modules, cache_names, selected_beam, cur_beam_size)
        visual = self._expand_visual(visual, cur_beam_size, selected_beam)

        self.seq_logprob = selected_logprob.unsqueeze(-1)
        self.seq_mask = torch.gather(self.seq_mask, 1, selected_beam.unsqueeze(-1))

        if outputs is None:
            outputs = selected_words.unsqueeze(-1)
        else:

            outputs = torch.gather(outputs, 1, selected_beam.unsqueeze(-1).expand_as(outputs))
            outputs = torch.cat([outputs, selected_words.unsqueeze(-1)], dim=2)


        if return_probs:
            if t == 0:
                self.all_log_probs.append(word_logprob.expand((self.b_s, self.beam_size, -1)).unsqueeze(2))
            else:
                self.all_log_probs.append(word_logprob.unsqueeze(2))

        this_word_logprob = torch.gather(word_logprob, 1,
                                         selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                            word_logprob.shape[-1]))
        this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))

        if self.log_probs is None:
            self.log_probs = this_word_logprob
        else:
            self.log_probs = torch.gather(self.log_probs, 1, selected_beam.unsqueeze(-1).expand_as(self.log_probs))
            self.log_probs = torch.cat([self.log_probs, this_word_logprob], dim=-1)

        self.selected_words = selected_words.view(-1, 1)

        return visual, outputs
