"""Tests for callbacks in training.py"""

from functools import partial
import pickle
from unittest.mock import Mock
from unittest.mock import patch
from unittest.mock import call
from copy import deepcopy

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import torch
from torch.utils.data import TensorDataset

from skorch.helper import predefined_split


class TestCheckpoint:
    @pytest.fixture
    def checkpoint_cls(self):
        from skorch.callbacks import Checkpoint
        return Checkpoint

    @pytest.fixture
    def save_params_mock(self):
        with patch('skorch.NeuralNet.save_params') as mock:
            yield mock

    @pytest.fixture(params=['torch', 'safetensors'])
    def use_safetensors(self, request):
        return request.param == 'safetensors'

    @pytest.fixture
    def pickle_dump_mock(self):
        with patch('pickle.dump') as mock:
            yield mock

    @pytest.fixture
    def net_cls(self):
        """very simple network that trains for 10 epochs"""
        from skorch import NeuralNetRegressor
        from skorch.toy import make_regressor

        module_cls = make_regressor(
            input_units=1,
            num_hidden=0,
            output_units=1,
        )

        return partial(
            NeuralNetRegressor,
            module=module_cls,
            max_epochs=10,
            batch_size=10)

    @pytest.fixture(scope='module')
    def data(self):
        # have 10 examples so we can do a nice CV split
        X = np.zeros((10, 1), dtype='float32')
        y = np.zeros((10, 1), dtype='float32')
        return X, y

    def test_init_with_wrong_kwarg_name_raises(self, checkpoint_cls):
        checkpoint_cls(f_foobar='foobar.pt').initialize()  # works
        msg = "Checkpoint got an unexpected argument 'foobar', did you mean 'f_foobar'?"
        with pytest.raises(TypeError, match=msg):
            checkpoint_cls(foobar='foobar.pt').initialize()

    def test_init_with_f_params_and_f_module_raises(self, checkpoint_cls):
        msg = "Checkpoint called with both f_params and f_module, please choose one"
        with pytest.raises(TypeError, match=msg):
            checkpoint_cls(f_module='weights.pt', f_params='params.pt').initialize()

    def test_init_with_f_optimizer_and_safetensors_raises(self, checkpoint_cls):
        msg = (
            "Cannot save optimizer state when using safetensors, "
            "please set f_optimizer=None or don't use safetensors."
        )
        with pytest.raises(ValueError, match=msg):
            checkpoint_cls(f_optimizer='optimizer.safetensors', use_safetensors=True)

    def test_none_monitor_saves_always(
            self, save_params_mock, net_cls, checkpoint_cls, data):
        sink = Mock()
        net = net_cls(callbacks=[
            checkpoint_cls(monitor=None, sink=sink,
                           event_name='event_another'),
        ])
        net.fit(*data)

        assert save_params_mock.call_count == 4 * len(net.history)
        assert sink.call_count == len(net.history)
        assert all((x is True) for x in net.history[:, 'event_another'])

    @pytest.mark.parametrize('message,files', [
        ('Unable to save module state to params.pt, '
         'Exception: encoding error',
         {
             'f_params': 'params.pt',
             'f_optimizer': None,
             'f_criterion': None,
             'f_history': None,
         }),
        ('Unable to save optimizer state to optimizer.pt, '
         'Exception: encoding error',
         {
             'f_params': None,
             'f_optimizer': 'optimizer.pt',
             'f_criterion': None,
             'f_history': None,
         }),
        ('Unable to save criterion state to criterion.pt, '
         'Exception: encoding error',
         {
             'f_params': None,
             'f_optimizer': None,
             'f_criterion': 'criterion.pt',
             'f_history': None,
         }),
        ('Unable to save history to history.json, '
         'Exception: encoding error',
         {
             'f_params': None,
             'f_optimizer': None,
             'f_criterion': None,
             'f_history': 'history.json',
         }),
    ])
    def test_outputs_to_sink_when_save_params_errors(
            self, save_params_mock, net_cls, checkpoint_cls, data,
            message, files):
        sink = Mock()
        save_params_mock.side_effect = Exception('encoding error')
        net = net_cls(callbacks=[
            checkpoint_cls(monitor=None, sink=sink, **files)
        ])
        net.fit(*data)

        assert save_params_mock.call_count == len(net.history)
        assert sink.call_count == 2*len(net.history)
        save_error_messages = [call(message)] * len(net.history)
        sink.assert_has_calls(save_error_messages, any_order=True)

    @pytest.mark.parametrize('f_name, mode', [
        ('f_params', 'w'),
        ('f_optimizer', 'w'),
        ('f_criterion', 'w'),
        ('f_history', 'w'),
        ('f_pickle', 'wb')
    ])
    def test_init_with_dirname_and_file_like_object_error(
            self, checkpoint_cls, tmpdir, f_name, mode):
        from skorch.exceptions import SkorchException

        skorch_dir = tmpdir.mkdir("skorch")
        exp_dir = skorch_dir.join("exp1")
        f = skorch_dir.join(f_name + ".pt")

        with f.open(mode) as fp:
            with pytest.raises(SkorchException) as e:
                checkpoint_cls(**{f_name: fp}, dirname=str(exp_dir))
        expected = "dirname can only be used when f_* are strings"
        assert str(e.value) == expected

    @pytest.mark.parametrize('f_name, mode', [
        ('f_params', 'w'),
        ('f_optimizer', 'w'),
        ('f_criterion', 'w'),
        ('f_history', 'w'),
        ('f_pickle', 'wb')
    ])
    def test_initialize_with_dirname_and_file_like_object_error(
            self, checkpoint_cls, tmpdir, f_name, mode):
        from skorch.exceptions import SkorchException

        skorch_dir = tmpdir.mkdir("skorch")
        exp_dir = skorch_dir.join("exp1")
        f = skorch_dir.join(f_name + ".pt")

        with f.open(mode) as fp:
            with pytest.raises(SkorchException) as e:
                cp = checkpoint_cls(dirname=str(exp_dir))
                setattr(cp, f_name, fp)
                cp.initialize()
        expected = "dirname can only be used when f_* are strings"
        assert str(e.value) == expected

    def test_default_without_validation_raises_meaningful_error(
            self, net_cls, checkpoint_cls, data):
        net = net_cls(
            callbacks=[
                checkpoint_cls(),
            ],
            train_split=None
        )
        from skorch.exceptions import SkorchException

        msg_expected = (
            r"Key 'valid_loss_best' was not found in history. "
            r"Make sure you have validation data if you use "
            r"validation scores for checkpointing."
        )
        with pytest.raises(SkorchException, match=msg_expected):
            net.fit(*data)

    def test_string_monitor_and_formatting(
            self, save_params_mock, net_cls, checkpoint_cls, data):
        def epoch_3_scorer(net, *_):
            return 1 if net.history[-1, 'epoch'] == 3 else 0

        from skorch.callbacks import EpochScoring
        scoring = EpochScoring(
            scoring=epoch_3_scorer, on_train=True, lower_is_better=False)

        sink = Mock()
        cb = checkpoint_cls(
            monitor='epoch_3_scorer_best',
            f_params='model_{last_epoch[epoch]}_{net.max_epochs}.pt',
            f_optimizer='optimizer_{last_epoch[epoch]}_{net.max_epochs}.pt',
            f_criterion='criterion_{last_epoch[epoch]}_{net.max_epochs}.pt',
            sink=sink)
        net = net_cls(callbacks=[
            ('my_score', scoring), cb
        ])
        net.fit(*data)

        assert save_params_mock.call_count == 8
        assert cb.get_formatted_files(net) == {
            'f_params': 'model_3_10.pt',
            'f_optimizer': 'optimizer_3_10.pt',
            'f_criterion': 'criterion_3_10.pt',
            'f_history': 'history.json',
            'f_pickle': None
        }
        save_params_mock.assert_has_calls(
            [
                # params is turned into module
                call(f_module='model_1_10.pt', use_safetensors=False),
                call(f_optimizer='optimizer_1_10.pt', use_safetensors=False),
                call(f_criterion='criterion_1_10.pt', use_safetensors=False),
                call(f_history='history.json', use_safetensors=False),
                # params is turned into module
                call(f_module='model_3_10.pt', use_safetensors=False),
                call(f_optimizer='optimizer_3_10.pt', use_safetensors=False),
                call(f_criterion='criterion_3_10.pt', use_safetensors=False),
                call(f_history='history.json', use_safetensors=False),
            ],
            any_order=True,
        )
        assert sink.call_count == 2
        # The first epoch will always be saved. `epoch_3_scorer` returns 1 at
        # epoch 3, which will trigger another checkpoint. For all other epochs
        # `epoch_3_scorer` returns 0, which does not trigger a checkpoint.
        assert [True, False, True] + [False] * 7 == net.history[:, 'event_cp']

    def test_save_all_targets(
            self, save_params_mock, pickle_dump_mock,
            net_cls, checkpoint_cls, data, use_safetensors):
        kwargs = dict(
            monitor=None,
            f_params='params.pt',
            f_pickle='model.pkl',
            f_optimizer='optimizer.pt',
            f_criterion='criterion.pt',
            f_history='history.json',
            use_safetensors=use_safetensors,
        )
        if use_safetensors:
            # safetensors cannot safe optimizers
            kwargs['f_optimizer'] = None
        net = net_cls(callbacks=[checkpoint_cls(**kwargs)])
        net.fit(*data)

        if use_safetensors:
            # no optimizer
            assert save_params_mock.call_count == 3 * len(net.history)
        else:
            assert save_params_mock.call_count == 4 * len(net.history)
        assert pickle_dump_mock.call_count == len(net.history)

        kwargs = {'use_safetensors': use_safetensors}
        calls_expected =             [
            call(f_module='params.pt', **kwargs),  # params is turned into module
            call(f_criterion='criterion.pt', **kwargs),
            call(f_history='history.json', **kwargs),
        ]
        if not use_safetensors:
            # safetensors cannot safe optimizers
            calls_expected.append(call(f_optimizer='optimizer.pt', **kwargs))
        save_params_mock.assert_has_calls(
            calls_expected * len(net.history),
            any_order=True,
        )

    def test_save_all_targets_with_prefix(
            self, save_params_mock, pickle_dump_mock,
            net_cls, checkpoint_cls, data, use_safetensors):

        kwargs = dict(
            monitor=None,
            f_params='params.pt',
            f_pickle='model.pkl',
            f_optimizer='optimizer.pt',
            f_criterion='criterion.pt',
            f_history='history.json',
            use_safetensors=use_safetensors,
            fn_prefix="exp1_",
        )
        if use_safetensors:
            # safetensors cannot safe optimizers
            kwargs['f_optimizer'] = None
        cp = checkpoint_cls(**kwargs)
        net = net_cls(callbacks=[cp])
        net.fit(*data)

        assert cp.f_history_ == "exp1_history.json"
        if use_safetensors:
            assert save_params_mock.call_count == 3 * len(net.history)
        else:
            assert save_params_mock.call_count == 4 * len(net.history)
        assert pickle_dump_mock.call_count == len(net.history)

        kwargs = {'use_safetensors': use_safetensors}
        calls_expected = [
            call(f_module='exp1_params.pt', **kwargs),
            call(f_criterion='exp1_criterion.pt', **kwargs),
            call(f_history='exp1_history.json', **kwargs),
        ]
        if not use_safetensors:
            # safetensors cannot safe optimizers
            calls_expected.append(call(f_optimizer='exp1_optimizer.pt', **kwargs))
        save_params_mock.assert_has_calls(
            calls_expected * len(net.history),
            any_order=True,
        )

    def test_save_all_targets_with_prefix_and_dirname(
            self, save_params_mock, pickle_dump_mock,
            net_cls, checkpoint_cls, data, tmpdir, use_safetensors):

        skorch_dir = tmpdir.mkdir('skorch').join('exp1')

        kwargs = dict(
            monitor=None,
            f_params='params.pt',
            f_history='history.json',
            f_pickle='model.pkl',
            f_optimizer='optimizer.pt',
            f_criterion='criterion.pt',
            fn_prefix="unet_",
            dirname=str(skorch_dir),
            use_safetensors=use_safetensors,
        )
        if use_safetensors:
            # safetensors cannot safe optimizers
            kwargs['f_optimizer'] = None
        cp = checkpoint_cls(**kwargs)
        net = net_cls(callbacks=[cp])
        net.fit(*data)

        f_params = skorch_dir.join('unet_params.pt')
        f_optimizer = skorch_dir.join('unet_optimizer.pt')
        f_criterion = skorch_dir.join('unet_criterion.pt')
        f_history = skorch_dir.join('unet_history.json')

        assert cp.f_history_ == str(f_history)
        if use_safetensors:
            assert save_params_mock.call_count == 3 * len(net.history)
        else:
            assert save_params_mock.call_count == 4 * len(net.history)
        assert pickle_dump_mock.call_count == len(net.history)

        kwargs = {'use_safetensors': use_safetensors}
        calls_expected = [
            call(f_module=str(f_params), **kwargs),  # params is turned into module
            call(f_criterion=str(f_criterion), **kwargs),
            call(f_history=str(f_history), **kwargs),
        ]
        if not use_safetensors:
            calls_expected.append(call(f_optimizer=str(f_optimizer), **kwargs))
        save_params_mock.assert_has_calls(
            calls_expected * len(net.history),
            any_order=True,
        )
        assert skorch_dir.exists()

    def test_save_no_targets(
            self, save_params_mock, pickle_dump_mock,
            net_cls, checkpoint_cls, data):
        net = net_cls(callbacks=[
            checkpoint_cls(
                monitor=None,
                f_params=None,
                f_optimizer=None,
                f_criterion=None,
                f_history=None,
                f_pickle=None,
            ),
        ])
        net.fit(*data)

        assert save_params_mock.call_count == 0
        assert pickle_dump_mock.call_count == 0

    def test_warnings_when_monitor_appears_in_history(
            self, net_cls, checkpoint_cls, save_params_mock, data):
        net = net_cls(
            callbacks=[checkpoint_cls(monitor="valid_loss")],
            max_epochs=1)

        exp_warn = (
            "Checkpoint monitor parameter is set to 'valid_loss' and the "
            "history contains 'valid_loss_best'. Perhaps you meant to set the "
            "parameter to 'valid_loss_best'")

        with pytest.warns(UserWarning, match=exp_warn):
            net.fit(*data)
        assert save_params_mock.call_count == 4

    def test_save_custom_module(
            self, save_params_mock, module_cls, checkpoint_cls, data, use_safetensors
    ):
        # checkpointing custom modules works
        from skorch import NeuralNetRegressor

        class MyNet(NeuralNetRegressor):
            """Net with custom module"""
            def __init__(self, *args, mymodule=module_cls, **kwargs):
                self.mymodule = mymodule
                super().__init__(*args, **kwargs)

            def initialize_module(self, *args, **kwargs):
                super().initialize_module(*args, **kwargs)

                params = self.get_params_for('mymodule')
                self.mymodule_ = self.mymodule(**params)

                return self

        cp = checkpoint_cls(
            monitor=None,
            f_params=None,
            f_optimizer=None,
            f_criterion=None,
            f_history=None,
            f_mymodule='mymodule.pt',
            use_safetensors=use_safetensors,
        )
        net = MyNet(module_cls, callbacks=[cp])
        net.fit(*data)

        assert save_params_mock.call_count == 1 * len(net.history)
        kwargs = {'use_safetensors': use_safetensors}
        save_params_mock.assert_has_calls(
            [call(f_mymodule='mymodule.pt', **kwargs)] * len(net.history)
        )

    @pytest.fixture
    def load_params(self):
        import torch
        return torch.load

    @pytest.mark.parametrize('load_best_flag', [False, True])
    def test_automatically_load_checkpoint(
            self, net_cls, checkpoint_cls, data, tmp_path,
            load_params, load_best_flag,
    ):
        # checkpoint once at the beginning of training.
        # when restoring at the end of training, the parameters
        # of the net should not differ. If we do not restore
        # then the parameters must differ.
        path_cb = tmp_path / 'params_cb.pt'
        path_net = tmp_path / 'params_net.pt'

        def save_once_monitor(net):
            return len(net.history) == 1

        net = net_cls(
            max_epochs=3,
            callbacks=[
                checkpoint_cls(
                    monitor=save_once_monitor,
                    f_params=path_cb,
                    load_best=load_best_flag,
                ),
            ],
        )

        net.fit(*data)
        net.save_params(path_net)

        params_cb = load_params(path_cb)
        params_net = load_params(path_net)

        if load_best_flag:
            assert params_cb == params_net
        else:
            assert params_cb != params_net



class TestEarlyStopping:

    @pytest.fixture
    def early_stopping_cls(self):
        from skorch.callbacks import EarlyStopping
        return EarlyStopping

    @pytest.fixture
    def epoch_scoring_cls(self):
        from skorch.callbacks import EpochScoring
        return EpochScoring

    @pytest.fixture
    def net_clf_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture
    def broken_classifier_module(self, classifier_module):
        """Return a classifier that does not improve over time."""
        class BrokenClassifier(classifier_module.func):
            def forward(self, x):
                return super().forward(x) * 0 + 0.5
        return BrokenClassifier

    def test_typical_use_case_nonstop(
            self, net_clf_cls, classifier_module, classifier_data,
            early_stopping_cls):
        patience = 5
        max_epochs = 8
        early_stopping_cb = early_stopping_cls(patience=patience)

        net = net_clf_cls(
            classifier_module,
            callbacks=[
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) == max_epochs

    def test_weights_restore(
            self, net_clf_cls, classifier_module, classifier_data,
            early_stopping_cls):
        patience = 3
        max_epochs = 20
        seed = 1

        side_effect = []

        def sink(x):
            side_effect.append(x)

        early_stopping_cb = early_stopping_cls(
            patience=patience,
            sink=sink,
            load_best=True,
            monitor="valid_acc",
            lower_is_better=False,
        )

        # Split dataset to have a fixed validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            *classifier_data, random_state=seed)
        tr_dataset = TensorDataset(
            torch.as_tensor(X_tr).float(), torch.as_tensor(y_tr))
        val_dataset = TensorDataset(
            torch.as_tensor(X_val).float(), torch.as_tensor(y_val))

        # Fix the network once with early stoppping and fixed seed
        net1 = net_clf_cls(
            classifier_module,
            callbacks=[early_stopping_cb],
            max_epochs=max_epochs,
            train_split=predefined_split(val_dataset),
        )
        torch.manual_seed(seed)
        net1.fit(tr_dataset, y=None)

        # Check training was stopped before the end
        assert len(net1.history) < max_epochs

        # check correct output messages
        assert len(side_effect) == 2

        msg = side_effect[0]
        expected_msg = ("Stopping since valid_acc has not improved in "
                        "the last 3 epochs.")
        assert msg == expected_msg

        msg = side_effect[1]
        expected_msg = "Restoring best model from epoch "
        assert expected_msg in msg

        # Recompute validation loss and store it together with module weights
        y_proba = net1.predict_proba(val_dataset)
        es_weights = deepcopy(net1.module_.state_dict())
        es_loss = log_loss(y_val, y_proba)

        # Retrain same classifier without ES, using the best epochs number
        net2 = net_clf_cls(
            classifier_module,
            max_epochs=early_stopping_cb.best_epoch_,
            train_split=predefined_split(val_dataset),
        )
        torch.manual_seed(seed)
        net2.fit(tr_dataset, y=None)

        # Check that weights obtained match
        assert all(
            torch.equal(wi, wj)
            for wi, wj in zip(
                net2.module_.state_dict().values(),
                es_weights.values()
            )
        )

        # Check validation loss obtained match
        y_proba_2 = net2.predict_proba(val_dataset)
        assert es_loss == log_loss(y_val, y_proba_2)

        # Check best_model_weights_ is transformed into None when pickling
        del net1.callbacks[0].sink
        net1_pkl = pickle.dumps(net1)

        reloaded_net1 = pickle.loads(net1_pkl)
        assert reloaded_net1.callbacks[0].best_epoch_ == net1.callbacks[0].best_epoch_
        assert reloaded_net1.callbacks[0].best_model_weights_ is None

    def test_typical_use_case_stopping(
            self, net_clf_cls, broken_classifier_module, classifier_data,
            early_stopping_cls):
        patience = 5
        max_epochs = 8
        side_effect = []

        def sink(x):
            side_effect.append(x)

        early_stopping_cb = early_stopping_cls(patience=patience, sink=sink)

        net = net_clf_cls(
            broken_classifier_module,
            callbacks=[
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) == patience + 1 < max_epochs

        # check correct output message
        assert len(side_effect) == 1
        msg = side_effect[0]
        expected_msg = ("Stopping since valid_loss has not improved in "
                        "the last 5 epochs.")
        assert msg == expected_msg

    def test_custom_scoring_nonstop(
            self, net_clf_cls, classifier_module, classifier_data,
            early_stopping_cls, epoch_scoring_cls,
    ):
        lower_is_better = False
        scoring_name = 'valid_roc_auc'
        patience = 5
        max_epochs = 8
        scoring_mock = Mock(side_effect=list(range(2, 10)))
        scoring_cb = epoch_scoring_cls(
            scoring_mock, lower_is_better, name=scoring_name)
        early_stopping_cb = early_stopping_cls(
            patience=patience, lower_is_better=lower_is_better,
            monitor=scoring_name)

        net = net_clf_cls(
            classifier_module,
            callbacks=[
                scoring_cb,
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) == max_epochs

    def test_custom_scoring_stop(
            self, net_clf_cls, broken_classifier_module, classifier_data,
            early_stopping_cls, epoch_scoring_cls,
    ):
        lower_is_better = False
        scoring_name = 'valid_roc_auc'
        patience = 5
        max_epochs = 8
        scoring_cb = epoch_scoring_cls(
            'roc_auc', lower_is_better, name=scoring_name)
        early_stopping_cb = early_stopping_cls(
            patience=patience, lower_is_better=lower_is_better,
            monitor=scoring_name)

        net = net_clf_cls(
            broken_classifier_module,
            callbacks=[
                scoring_cb,
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) < max_epochs

    def test_stopping_big_absolute_threshold(
            self, net_clf_cls, classifier_module, classifier_data,
            early_stopping_cls):
        patience = 5
        max_epochs = 8
        early_stopping_cb = early_stopping_cls(patience=patience,
                                               threshold_mode='abs',
                                               threshold=0.1)

        net = net_clf_cls(
            classifier_module,
            callbacks=[
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) == patience + 1 < max_epochs

    def test_wrong_threshold_mode(
            self, net_clf_cls, classifier_module, classifier_data,
            early_stopping_cls):
        patience = 5
        max_epochs = 8
        early_stopping_cb = early_stopping_cls(
            patience=patience, threshold_mode='incorrect')
        net = net_clf_cls(
            classifier_module,
            callbacks=[
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )

        with pytest.raises(ValueError) as exc:
            net.fit(*classifier_data)

        expected_msg = "Invalid threshold mode: 'incorrect'"
        assert exc.value.args[0] == expected_msg



class TestParamMapper:

    @pytest.fixture
    def initializer(self):
        from skorch.callbacks import Initializer
        return Initializer

    @pytest.fixture
    def freezer(self):
        from skorch.callbacks import Freezer
        return Freezer

    @pytest.fixture
    def unfreezer(self):
        from skorch.callbacks import Unfreezer
        return Unfreezer

    @pytest.fixture
    def param_mapper(self):
        from skorch.callbacks import ParamMapper
        return ParamMapper

    @pytest.fixture
    def net_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.mark.parametrize('at', [0, -1])
    def test_subzero_at_fails(self, net_cls, classifier_module,
                              param_mapper, at):
        cb = param_mapper(patterns='*', at=at)
        net = net_cls(classifier_module, callbacks=[cb])
        with pytest.raises(ValueError):
            net.initialize()

    @pytest.mark.parametrize('mod_init', [False, True])
    @pytest.mark.parametrize('weight_pattern', [
        'sequential.*.weight',
        lambda name: name.startswith('sequential') and name.endswith('.weight'),
    ])
    def test_initialization_is_effective(self, net_cls, classifier_module,
                                         classifier_data, initializer,
                                         mod_init, weight_pattern):
        from torch.nn.init import constant_
        from skorch.utils import to_numpy

        module = classifier_module() if mod_init else classifier_module

        net = net_cls(
            module,
            lr=0,
            max_epochs=1,
            callbacks=[
                initializer(weight_pattern, partial(constant_, val=5)),
                initializer('sequential.3.bias', partial(constant_, val=10)),
            ])

        net.fit(*classifier_data)

        assert np.allclose(to_numpy(net.module_.sequential[0].weight), 5)
        assert np.allclose(to_numpy(net.module_.sequential[3].weight), 5)
        assert np.allclose(to_numpy(net.module_.sequential[3].bias), 10)

    @pytest.mark.parametrize('mod_init', [False, True])
    @pytest.mark.parametrize('mod_kwargs', [
        {},
        # Supply a module__ parameter so the model is forced
        # to re-initialize. Even then parameters should be
        # frozen correctly.
        {'module__hidden_units': 5},
    ])
    def test_freezing_is_effective(self, net_cls, classifier_module,
                                   classifier_data, freezer, mod_init,
                                   mod_kwargs):
        from skorch.utils import to_numpy

        module = classifier_module() if mod_init else classifier_module

        net = net_cls(
            module,
            max_epochs=2,
            callbacks=[
                freezer('sequential.*.weight'),
                freezer('sequential.3.bias'),
            ],
            **mod_kwargs)

        net.initialize()

        assert net.module_.sequential[0].weight.requires_grad
        assert net.module_.sequential[3].weight.requires_grad
        assert net.module_.sequential[0].bias.requires_grad
        assert net.module_.sequential[3].bias.requires_grad

        dense0_weight_pre = to_numpy(net.module_.sequential[0].weight).copy()
        dense1_weight_pre = to_numpy(net.module_.sequential[3].weight).copy()
        dense0_bias_pre = to_numpy(net.module_.sequential[0].bias).copy()
        dense1_bias_pre = to_numpy(net.module_.sequential[3].bias).copy()

        # use partial_fit to not re-initialize the module (weights)
        net.partial_fit(*classifier_data)

        dense0_weight_post = to_numpy(net.module_.sequential[0].weight).copy()
        dense1_weight_post = to_numpy(net.module_.sequential[3].weight).copy()
        dense0_bias_post = to_numpy(net.module_.sequential[0].bias).copy()
        dense1_bias_post = to_numpy(net.module_.sequential[3].bias).copy()

        assert not net.module_.sequential[0].weight.requires_grad
        assert not net.module_.sequential[3].weight.requires_grad
        assert net.module_.sequential[0].bias.requires_grad
        assert not net.module_.sequential[3].bias.requires_grad

        assert np.allclose(dense0_weight_pre, dense0_weight_post)
        assert np.allclose(dense1_weight_pre, dense1_weight_post)
        assert not np.allclose(dense0_bias_pre, dense0_bias_post)
        assert np.allclose(dense1_bias_pre, dense1_bias_post)

    def test_unfreezing_is_effective(self, net_cls, classifier_module,
                                     classifier_data, freezer, unfreezer):
        from skorch.utils import to_numpy

        net = net_cls(
            classifier_module,
            max_epochs=1,
            callbacks=[
                freezer('sequential.*.weight'),
                freezer('sequential.3.bias'),
                unfreezer('sequential.*.weight', at=2),
                unfreezer('sequential.3.bias', at=2),
            ])

        net.initialize()

        # epoch 1, freezing parameters
        net.partial_fit(*classifier_data)

        assert not net.module_.sequential[0].weight.requires_grad
        assert not net.module_.sequential[3].weight.requires_grad
        assert net.module_.sequential[0].bias.requires_grad
        assert not net.module_.sequential[3].bias.requires_grad

        dense0_weight_pre = to_numpy(net.module_.sequential[0].weight).copy()
        dense1_weight_pre = to_numpy(net.module_.sequential[3].weight).copy()
        dense0_bias_pre = to_numpy(net.module_.sequential[0].bias).copy()
        dense1_bias_pre = to_numpy(net.module_.sequential[3].bias).copy()

        # epoch 2, unfreezing parameters
        net.partial_fit(*classifier_data)

        assert net.module_.sequential[0].weight.requires_grad
        assert net.module_.sequential[3].weight.requires_grad
        assert net.module_.sequential[0].bias.requires_grad
        assert net.module_.sequential[3].bias.requires_grad

        # epoch 3, modifications should have been made
        net.partial_fit(*classifier_data)

        dense0_weight_post = to_numpy(net.module_.sequential[0].weight).copy()
        dense1_weight_post = to_numpy(net.module_.sequential[3].weight).copy()
        dense0_bias_post = to_numpy(net.module_.sequential[0].bias).copy()
        dense1_bias_post = to_numpy(net.module_.sequential[3].bias).copy()

        assert not np.allclose(dense0_weight_pre, dense0_weight_post)
        assert not np.allclose(dense1_weight_pre, dense1_weight_post)
        assert not np.allclose(dense0_bias_pre, dense0_bias_post)
        assert not np.allclose(dense1_bias_pre, dense1_bias_post)


    def test_schedule_is_effective(self, net_cls, classifier_module,
                                   classifier_data, param_mapper):
        from skorch.utils import to_numpy, noop
        from skorch.utils import freeze_parameter, unfreeze_parameter

        def schedule(net):
            if len(net.history) == 1:
                return freeze_parameter
            elif len(net.history) == 2:
                return unfreeze_parameter
            return noop

        net = net_cls(
            classifier_module,
            max_epochs=1,
            callbacks=[
                param_mapper(
                    ['sequential.*.weight', 'sequential.3.bias'],
                    schedule=schedule,
                ),
            ])

        net.initialize()

        # epoch 1, freezing parameters
        net.partial_fit(*classifier_data)

        assert not net.module_.sequential[0].weight.requires_grad
        assert not net.module_.sequential[3].weight.requires_grad
        assert net.module_.sequential[0].bias.requires_grad
        assert not net.module_.sequential[3].bias.requires_grad

        dense0_weight_pre = to_numpy(net.module_.sequential[0].weight).copy()
        dense1_weight_pre = to_numpy(net.module_.sequential[3].weight).copy()
        dense0_bias_pre = to_numpy(net.module_.sequential[0].bias).copy()
        dense1_bias_pre = to_numpy(net.module_.sequential[3].bias).copy()

        # epoch 2, unfreezing parameters
        net.partial_fit(*classifier_data)

        assert net.module_.sequential[0].weight.requires_grad
        assert net.module_.sequential[3].weight.requires_grad
        assert net.module_.sequential[0].bias.requires_grad
        assert net.module_.sequential[3].bias.requires_grad

        # epoch 3, modifications should have been made
        net.partial_fit(*classifier_data)

        dense0_weight_post = to_numpy(net.module_.sequential[0].weight).copy()
        dense1_weight_post = to_numpy(net.module_.sequential[3].weight).copy()
        dense0_bias_post = to_numpy(net.module_.sequential[0].bias).copy()
        dense1_bias_post = to_numpy(net.module_.sequential[3].bias).copy()

        assert not np.allclose(dense0_weight_pre, dense0_weight_post)
        assert not np.allclose(dense1_weight_pre, dense1_weight_post)
        assert not np.allclose(dense0_bias_pre, dense0_bias_post)
        assert not np.allclose(dense1_bias_pre, dense1_bias_post)


class TestLoadInitState:
    @pytest.fixture(params=['torch', 'safetensors'])
    def use_safetensors(self, request):
        return request.param == 'safetensors'

    @pytest.fixture
    def checkpoint_cls(self):
        from skorch.callbacks import Checkpoint
        return Checkpoint

    @pytest.fixture
    def loadinitstate_cls(self):
        from skorch.callbacks import LoadInitState
        return LoadInitState

    @pytest.fixture
    def net_cls(self):
        """very simple network that trains for 10 epochs"""
        from skorch import NeuralNetRegressor
        from skorch.toy import make_regressor

        module_cls = make_regressor(
            input_units=1,
            num_hidden=0,
            output_units=1,
        )

        return partial(
            NeuralNetRegressor,
            module=module_cls,
            max_epochs=10,
            batch_size=10)

    @pytest.fixture(scope='module')
    def data(self):
        # have 10 examples so we can do a nice CV split
        X = np.zeros((10, 1), dtype='float32')
        y = np.zeros((10, 1), dtype='float32')
        return X, y

    def test_load_initial_state(
            self, checkpoint_cls, net_cls, loadinitstate_cls,
            data, tmpdir, use_safetensors):
        skorch_dir = tmpdir.mkdir('skorch')
        f_params = skorch_dir.join('params.pt')
        f_optimizer = skorch_dir.join('optimizer.pt')
        f_criterion = skorch_dir.join('criterion.pt')
        f_history = skorch_dir.join('history.json')

        kwargs = dict(
            monitor=None,
            f_params=str(f_params),
            f_optimizer=str(f_optimizer),
            f_criterion=str(f_criterion),
            f_history=str(f_history),
            use_safetensors=use_safetensors,
        )
        if use_safetensors:
            # safetensors cannot safe optimizers
            kwargs['f_optimizer'] = None
        cp = checkpoint_cls(**kwargs)
        load_init_state = loadinitstate_cls(cp, use_safetensors=use_safetensors)
        net = net_cls(callbacks=[cp, load_init_state])
        net.fit(*data)

        assert f_params.exists()
        assert f_criterion.exists()
        assert f_history.exists()
        if not use_safetensors:
            # safetensors cannot safe optimizers
            assert f_optimizer.exists()

        assert len(net.history) == 10
        del net

        new_net = net_cls(callbacks=[cp, load_init_state])
        new_net.fit(*data)

        assert len(new_net.history) == 20

    def test_load_initial_state_custom_scoring(
            self, checkpoint_cls, net_cls, loadinitstate_cls,
            data, tmpdir, use_safetensors):
        def epoch_3_scorer(net, *_):
            return 1 if net.history[-1, 'epoch'] == 3 else 0

        from skorch.callbacks import EpochScoring
        scoring = EpochScoring(
            scoring=epoch_3_scorer, on_train=True, lower_is_better=False)

        skorch_dir = tmpdir.mkdir('skorch')
        f_params = skorch_dir.join(
            'model_epoch_{last_epoch[epoch]}.pt')
        f_optimizer = skorch_dir.join(
            'optimizer_epoch_{last_epoch[epoch]}.pt')
        f_criterion = skorch_dir.join(
            'criterion_epoch_{last_epoch[epoch]}.pt')
        f_history = skorch_dir.join(
            'history.json')

        kwargs = dict(
            monitor='epoch_3_scorer_best',
            f_params=str(f_params),
            f_optimizer=str(f_optimizer),
            f_criterion=str(f_criterion),
            f_history=str(f_history),
            use_safetensors=use_safetensors,
        )
        if use_safetensors:
            # safetensors cannot safe optimizers
            kwargs['f_optimizer'] = None
        cp = checkpoint_cls(**kwargs)
        load_init_state = loadinitstate_cls(cp, use_safetensors=use_safetensors)
        net = net_cls(callbacks=[load_init_state, scoring, cp])

        net.fit(*data)

        assert skorch_dir.join('model_epoch_3.pt').exists()
        assert skorch_dir.join('criterion_epoch_3.pt').exists()
        assert skorch_dir.join('history.json').exists()
        if not use_safetensors:
            # safetensors cannot safe optimizers
            assert skorch_dir.join('optimizer_epoch_3.pt').exists()

        assert len(net.history) == 10
        del net

        new_net = net_cls(callbacks=[load_init_state, scoring, cp])
        new_net.fit(*data)

        # new_net starts from the best epoch of the first run
        # the best epcoh of the previous run was at epoch 3
        # the second run went through 10 epochs, thus
        # 3 + 10 = 13
        assert len(new_net.history) == 13
        assert new_net.history[:, 'event_cp'] == [
            True, False, True] + [False] * 10


class TestTrainEndCheckpoint:
    @pytest.fixture(params=['torch', 'safetensors'])
    def use_safetensors(self, request):
        return request.param == 'safetensors'

    @pytest.fixture
    def trainendcheckpoint_cls(self):
        from skorch.callbacks import TrainEndCheckpoint
        return TrainEndCheckpoint

    @pytest.fixture
    def save_params_mock(self):
        with patch('skorch.NeuralNet.save_params') as mock:
            yield mock

    @pytest.fixture
    def net_cls(self):
        """very simple network that trains for 10 epochs"""
        from skorch import NeuralNetRegressor
        from skorch.toy import make_regressor

        module_cls = make_regressor(
            input_units=1,
            num_hidden=0,
            output_units=1,
        )

        return partial(
            NeuralNetRegressor,
            module=module_cls,
            max_epochs=10,
            batch_size=10)

    @pytest.fixture(scope='module')
    def data(self):
        # have 10 examples so we can do a nice CV split
        X = np.zeros((10, 1), dtype='float32')
        y = np.zeros((10, 1), dtype='float32')
        return X, y

    def test_init_with_wrong_kwarg_name_raises(self, trainendcheckpoint_cls):
        trainendcheckpoint_cls(f_foobar='foobar.pt').initialize()  # works
        msg = ("TrainEndCheckpoint got an unexpected argument 'foobar', "
               "did you mean 'f_foobar'?")
        with pytest.raises(TypeError, match=msg):
            trainendcheckpoint_cls(foobar='foobar.pt').initialize()

    def test_init_with_f_params_and_f_module_raises(self, trainendcheckpoint_cls):
        msg = "Checkpoint called with both f_params and f_module, please choose one"
        with pytest.raises(TypeError, match=msg):
            trainendcheckpoint_cls(
                f_module='weights.pt', f_params='params.pt').initialize()

    def test_init_with_f_optimizer_and_safetensors_raises(self, trainendcheckpoint_cls):
        msg = (
            "Cannot save optimizer state when using safetensors, "
            "please set f_optimizer=None or don't use safetensors."
        )
        with pytest.raises(ValueError, match=msg):
            trainendcheckpoint_cls(
                f_optimizer='optimizer.safetensors', use_safetensors=True
            )

    def test_saves_at_end(
            self,
            save_params_mock,
            net_cls,
            trainendcheckpoint_cls,
            data,
            use_safetensors,
    ):
        sink = Mock()
        kwargs = dict(
            sink=sink,
            dirname='exp1',
            fn_prefix='train_end_',
            use_safetensors=use_safetensors,
        )
        if use_safetensors:
            # safetensors cannot safe optimizers
            kwargs['f_optimizer'] = None
        net = net_cls(callbacks=[trainendcheckpoint_cls(**kwargs)])
        net.fit(*data)

        if use_safetensors:
            # safetensors cannot safe optimizers
            assert save_params_mock.call_count == 3
        else:
            assert save_params_mock.call_count == 4
        assert sink.call_args == call("Final checkpoint triggered")

        kwargs = {'use_safetensors': use_safetensors}
        calls_expected = [
            # params is turned into module
            call(f_module='exp1/train_end_params.pt', **kwargs),
            call(f_criterion='exp1/train_end_criterion.pt', **kwargs),
            call(f_history='exp1/train_end_history.json', **kwargs),
        ]
        if not use_safetensors:
            calls_expected.append(
                call(f_optimizer='exp1/train_end_optimizer.pt', **kwargs)
            )
        save_params_mock.assert_has_calls(
            calls_expected,
            any_order=True,
        )

    def test_saves_at_end_with_custom_formatting(
            self,
            save_params_mock,
            net_cls,
            trainendcheckpoint_cls,
            data,
            use_safetensors,
    ):
        sink = Mock()
        kwargs = dict(
            sink=sink,
            dirname='exp1',
            f_params='model_{last_epoch[epoch]}.pt',
            f_optimizer='optimizer_{last_epoch[epoch]}.pt',
            f_criterion='criterion_{last_epoch[epoch]}.pt',
            fn_prefix='train_end_',
            use_safetensors=use_safetensors,
        )
        if use_safetensors:
            # safetensors cannot safe optimizers
            kwargs['f_optimizer'] = None
        net = net_cls(callbacks=[trainendcheckpoint_cls(**kwargs)])
        net.fit(*data)

        if use_safetensors:
            # safetensors cannot safe optimizers
            assert save_params_mock.call_count == 3
        else:
            assert save_params_mock.call_count == 4
        assert sink.call_args == call("Final checkpoint triggered")

        kwargs = {'use_safetensors': use_safetensors}
        calls_expected = [
                # params is turned into module
            call(f_module='exp1/train_end_model_10.pt', **kwargs),
            call(f_criterion='exp1/train_end_criterion_10.pt', **kwargs),
            call(f_history='exp1/train_end_history.json', **kwargs),
        ]
        if not use_safetensors:
            calls_expected.append(
                call(f_optimizer='exp1/train_end_optimizer_10.pt', **kwargs),
            )
        save_params_mock.assert_has_calls(
            calls_expected,
            any_order=True,
        )

    def test_cloneable(self, trainendcheckpoint_cls):
        # reproduces bug #459
        cp = trainendcheckpoint_cls()
        clone(cp)  # does not raise

    def test_train_end_with_load_init(self, trainendcheckpoint_cls, net_cls, data):
        # test for https://github.com/skorch-dev/skorch/issues/528
        # Check that the initial state is indeed loaded from the checkpoint.
        from skorch.callbacks import LoadInitState
        from sklearn.metrics import mean_squared_error

        X, y = data
        cp = trainendcheckpoint_cls()
        net = net_cls(callbacks=[cp], max_epochs=3, lr=0.1).initialize()
        score_before = mean_squared_error(y, net.predict(X))
        net.partial_fit(X, y)
        score_after = mean_squared_error(y, net.predict(X))

        # make sure the net learned at all
        assert score_after < score_before

        net_new = net_cls(callbacks=[LoadInitState(cp)], max_epochs=0)
        net_new.fit(X, y)
        score_loaded = mean_squared_error(y, net_new.predict(X))

        # the same score as after the end of training of the initial
        # net should be obtained
        assert np.isclose(score_loaded, score_after)

    def test_save_custom_module(
            self,
            save_params_mock,
            module_cls,
            trainendcheckpoint_cls,
            data,
            use_safetensors,
    ):
        # checkpointing custom modules works
        from skorch import NeuralNetRegressor

        class MyNet(NeuralNetRegressor):
            """Net with custom module"""
            def __init__(self, *args, mymodule=module_cls, **kwargs):
                self.mymodule = mymodule
                super().__init__(*args, **kwargs)

            def initialize_module(self, *args, **kwargs):
                super().initialize_module(*args, **kwargs)

                params = self.get_params_for('mymodule')
                self.mymodule_ = self.mymodule(**params)

                return self

        cp = trainendcheckpoint_cls(
            f_params=None,
            f_optimizer=None,
            f_criterion=None,
            f_history=None,
            f_mymodule='mymodule.pt',
            use_safetensors=use_safetensors,
        )
        net = MyNet(module_cls, callbacks=[cp])
        net.fit(*data)

        kwargs = {'use_safetensors': use_safetensors}
        assert save_params_mock.call_count == 1
        save_params_mock.assert_has_calls(
            [call(f_mymodule='train_end_mymodule.pt', **kwargs)]
        )

    def test_pickle_uninitialized_callback(self, trainendcheckpoint_cls):
        # isuue 773
        cp = trainendcheckpoint_cls()
        # does not raise
        s = pickle.dumps(cp)
        pickle.loads(s)

    def test_pickle_initialized_callback(self, trainendcheckpoint_cls):
        # issue 773
        cp = trainendcheckpoint_cls().initialize()
        # does not raise
        s = pickle.dumps(cp)
        pickle.loads(s)


class TestInputShapeSetter:

    @pytest.fixture
    def module_cls(self):
        import torch

        class Module(torch.nn.Module):
            def __init__(self, input_dim=3):
                super().__init__()
                self.layer = torch.nn.Linear(input_dim, 2)
            def forward(self, X):
                return self.layer(X)

        return Module

    @pytest.fixture
    def net_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture
    def input_shape_setter_cls(self):
        from skorch.callbacks import InputShapeSetter
        return InputShapeSetter

    def generate_data(self, n_input):
        from sklearn.datasets import make_classification
        X, y = make_classification(
            1000,
            n_input,
            n_informative=n_input,
            n_redundant=0,
            random_state=0,
        )
        return X.astype(np.float32), y

    @pytest.fixture
    def data_fixed(self):
        return self.generate_data(n_input=10)

    @pytest.fixture(params=[2, 10, 20])
    def data_parametrized(self, request):
        return self.generate_data(n_input=request.param)

    def test_shape_set(
        self, net_cls, module_cls, input_shape_setter_cls, data_parametrized,
    ):
        net = net_cls(module_cls, max_epochs=2, callbacks=[
            input_shape_setter_cls(),
        ])

        X, y = data_parametrized
        n_input = X.shape[1]
        net.fit(X, y)

        assert net.module_.layer.in_features == n_input

    def test_one_dimensional_x_raises(
        self, net_cls, module_cls, input_shape_setter_cls,
    ):
        net = net_cls(module_cls, max_epochs=2, callbacks=[
            input_shape_setter_cls(),
        ])

        X, y = np.zeros(10), np.zeros(10)

        with pytest.raises(ValueError) as e:
            net.fit(X, y)

        assert (
            "Expected at least two-dimensional input data for X. "
            "If your data is one-dimensional, please use the `input_dim_fn` "
            "parameter to infer the correct input shape."
            ) in str(e)

    def test_shape_set_using_fn(
        self, net_cls, module_cls, input_shape_setter_cls, data_parametrized,
    ):
        fn_calls = 0

        def input_dim_fn(X):
            nonlocal fn_calls
            fn_calls += 1
            return X.shape[1]

        net = net_cls(module_cls, max_epochs=2, callbacks=[
            input_shape_setter_cls(input_dim_fn=input_dim_fn),
        ])

        X, y = data_parametrized
        n_input = X.shape[1]
        net.fit(X, y)

        assert net.module_.layer.in_features == n_input
        assert fn_calls == 1

    def test_parameter_name(
        self, net_cls, input_shape_setter_cls, data_parametrized,
    ):
        class MyModule(torch.nn.Module):
            def __init__(self, other_input_dim=22):
                super().__init__()
                self.layer = torch.nn.Linear(other_input_dim, 2)
            def forward(self, X):
                return self.layer(X)

        net = net_cls(MyModule, max_epochs=2, callbacks=[
            input_shape_setter_cls(param_name='other_input_dim'),
        ])

        X, y = data_parametrized
        n_input = X.shape[1]
        net.fit(X, y)

        assert net.module_.layer.in_features == n_input

    def test_module_name(
        self, net_cls, module_cls, input_shape_setter_cls, data_parametrized,
    ):
        class MyNet(net_cls):
            def initialize_module(self):
                kwargs = self.get_params_for('module')
                self.module_ = self.module(**kwargs)

                kwargs = self.get_params_for('module2')
                self.module2_ = self.module(**kwargs)

        net = MyNet(
            module=module_cls,
            max_epochs=2,
            callbacks=[
                input_shape_setter_cls(module_name='module'),
                input_shape_setter_cls(module_name='module2'),
            ],
        )

        X, y = data_parametrized
        n_input = X.shape[1]
        net.fit(X, y)

        assert net.module_.layer.in_features == n_input
        assert net.module2_.layer.in_features == n_input

    def test_no_module_reinit_when_already_correct(
        self, net_cls, module_cls, input_shape_setter_cls, data_fixed,
    ):
        with patch('skorch.classifier.NeuralNetClassifier.initialize_module',
                   side_effect=net_cls.initialize_module, autospec=True):
            net = net_cls(
                module_cls, max_epochs=2, callbacks=[input_shape_setter_cls()],

                # set the input dim to the correct shape beforehand
                module__input_dim=data_fixed[0].shape[-1],
            )

            net.fit(*data_fixed)

            # first initialization due to `initialize()` but not
            # a second one since the input shape is already correct.
            assert net.initialize_module.call_count == 1

    def test_no_module_reinit_partial_fit(
        self, net_cls, module_cls, input_shape_setter_cls, data_fixed,
    ):
        with patch('skorch.classifier.NeuralNetClassifier.initialize_module',
                   side_effect=net_cls.initialize_module, autospec=True):
            net = net_cls(
                module_cls, max_epochs=2, callbacks=[input_shape_setter_cls()],
            )

            net.fit(*data_fixed)
            # first initialization due to `initialize()`, second
            # by setting the input dimension in `on_train_begin`
            assert net.initialize_module.call_count == 2

            net.partial_fit(*data_fixed)
            # no re-initialization when there was no change in
            # input dimension.
            assert net.initialize_module.call_count == 2
