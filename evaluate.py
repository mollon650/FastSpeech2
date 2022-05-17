import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset
import numpy as np
from model.modules import LengthRegulator
try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')
from torch.utils.mobile_optimizer import optimize_for_mobile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def evaluate(model, step, configs, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(6)]
    ################################################################
    # for batchs in loader:
    #     for batch in batchs:
    #         # export_onnx(model, batch)
    #         batch = to_device(batch, device)
    #         with torch.no_grad():
    #             # Forward
    #             output = model(*(batch[2:]))

    #             # Cal Loss
    #             losses = Loss(batch, output)

    #             for i in range(len(losses)):
    #                 loss_sums[i] += losses[i].item() * len(batch[0])
#####################################################################
    for i in range(32):
        raw_texts_np = np.load('./evalbin/'+str(i)+'raw_texts_np.npy')
        ids_np = np.load('./evalbin/'+str(i)+'ids_np.npy')
        speakers = np.load('./evalbin/'+str(i)+'speakers.npy')
        texts = np.load('./evalbin/'+str(i)+'texts.npy')
        text_lens = np.load('./evalbin/'+str(i)+'text_lens.npy')
        mels = np.load('./evalbin/'+str(i)+'mels.npy')
        mel_lens = np.load('./evalbin/'+str(i)+'mel_lens.npy')
        pitches = np.load('./evalbin/'+str(i)+'pitches.npy')
        energies = np.load('./evalbin/'+str(i)+'energies.npy')
        durations = np.load('./evalbin/'+str(i)+'durations.npy')
        ids = ids_np.tolist()
        raw_texts = raw_texts_np.tolist()
        
        batch =  (ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            125,
            mels,
            mel_lens,
            868,
            pitches,
            energies,
            durations,
        )
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(*(batch[2:]))

            # Cal Loss
            losses = Loss(batch, output)

            for i in range(len(losses)):
                loss_sums[i] += losses[i].item() * len(batch[0])
                
    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )

    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    print(dataset.tests_max_len, dataset.mels_max_len)
    return message

def export_onnx(model):
    input_names = ['speakers', 'texts', 'src_lens', 'max_src_len']
    output_names = ['output', 'postnet_output', 'p_predictions', 'e_predictions', 'log_d_predictions', 'd_rounded',
	                'src_masks', 'mel_masks', 'src_lens', 'mel_lens']
    dynamic_axes = {
		# "texts": {1: "texts_len"},
		"output": {1: "output_len"},
		"postnet_output": {1: "postnet_output_len"},
		"p_predictions": {1: "p_predictions_len"},
		"e_predictions": {1: "e_predictions_len"},
		"log_d_predictions": {1: "log_d_predictions_len"},
		"d_rounded": {1: "d_rounded_len"},
		"src_masks": {1: "src_masks_len"}
	}

    texts_len = 10
    speakers = torch.tensor([0]).to(device)
    texts = torch.randint(1, 200, (1, texts_len)).to(device)
    text_lens = torch.tensor([texts_len]).to(device)
    max_len = torch.from_numpy(np.array(texts_len)).to(device)
    torch.onnx.export(model, args=(speakers, texts, text_lens, max_len), f="./FastSpeech_2.onnx",
                        input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, opset_version=11)  

def export_pt_to_onnx():
    model = torch.jit.load('./xx.pt')
    model.eval()
    input_names = ['speakers', 'texts', 'src_lens', 'max_src_len']
    output_names = ['output', 'postnet_output', 'p_predictions', 'e_predictions', 'log_d_predictions', 'd_rounded',
	                'src_masks', 'mel_masks', 'src_lens', 'mel_lens']
    # dynamic_axes = {
	# 	# "texts": {1: "texts_len"},
	# 	"output": {1: "output_len"},
	# 	"postnet_output": {1: "postnet_output_len"},
	# 	"p_predictions": {1: "p_predictions_len"},
	# 	"e_predictions": {1: "e_predictions_len"},
	# 	"log_d_predictions": {1: "log_d_predictions_len"},
	# 	"d_rounded": {1: "d_rounded_len"},
	# 	"src_masks": {1: "src_masks_len"}
	# }

    texts_len = 10
    speakers = torch.tensor([0]).to(device)
    texts = torch.randint(1, 200, (1, texts_len)).to(device)
    text_lens = torch.tensor([texts_len]).to(device)
    max_len = torch.from_numpy(np.array(texts_len)).to(device)
    mels = torch.tensor(0)
    mel_lens = torch.tensor(0)
    max_mel_len = torch.tensor(0)
    p_targets = torch.tensor(0)
    e_targets = torch.tensor(0)
    d_targets = torch.tensor(0)
    p_control = torch.tensor(1.0)
    e_control = torch.tensor(1.0)
    d_control = torch.tensor(1.0)
    
    torch.onnx.export(model, args=(speakers, texts, text_lens, max_len), f="./FastSpeech_pt.onnx",
                        input_names=input_names, output_names=output_names, opset_version=10)

def trace_model(model):
    texts_len = 10
    speakers = torch.tensor([0]).to(device)
    texts = torch.randint(1, 200, (1, texts_len)).to(device)
    text_lens = torch.tensor([texts_len]).to(device)
    # max_len = torch.from_numpy(np.array(texts_len)).to(device)
    max_len:int = texts_len
    traced_script_module = torch.jit.script(model, example_inputs=(speakers, texts, text_lens, max_len))
    print(traced_script_module)
    opt_model = optimize_for_mobile(traced_script_module)
    traced_script_module.save("xx.pt")
    opt_model.save("opt_xx.pt")
    
def eval_onnx(model):
    ########################################################
    model.eval()
    texts_len = 10
    speakers = torch.tensor([0])
    input_tensor = torch.randn([1,3,224,224])
    texts = torch.randint(1, 200, (1, texts_len)).to(device)
    text_lens = torch.tensor([texts_len]).to(device)
    max_len = torch.from_numpy(np.array(texts_len)).to(device)
    output_file = './FastSpeech_2.onnx'
    # register_extra_symbolics(opset_version)
    # torch.onnx.export(
    #     model,
    #     input_tensor,
    #     './swin.onnx',
    #     export_params=True,
    #     keep_initializers_as_inputs=True,
    #     verbose=True,
    #     opset_version=11)
    print(f'Successfully exported ONNX model: {output_file}')

    # check by onnx
    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model)

    # check the numerical value
    # get pytorch output
    pytorch_result = model(speakers.to(device), texts.to(device), text_lens.to(device), max_len.to(device))

    # get onnx output
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [
        node.name for node in onnx_model.graph.initializer
    ]
    net_feed_input = list(set(input_all) - set(input_initializer))
    # assert len(net_feed_input) == 1
    sess = rt.InferenceSession(output_file)
    onnx_result = sess.run(
        None, {'src_lens': text_lens.detach().numpy(),
               'texts': texts.detach().numpy(),
               'max_src_len': max_len.detach().numpy()})
    # only compare part of results
    random_class = np.random.randint(pytorch_result.shape[1])
    assert np.allclose(
        pytorch_result[:, random_class], onnx_result[:, random_class]
    ), 'The outputs are different between Pytorch and ONNX'
    print('The numerical values are same between Pytorch and ONNX')
            ########################################################


def eval_pt(model):
    pt_model = torch.jit.load('./xx.pt')
    pt_model.eval()
    texts_len = 13
    speakers = torch.tensor([0])
    input_tensor = torch.randn([1,3,224,224])
    # texts = torch.randint(1, 200, (1, texts_len)).to(device)
    texts = torch.tensor([[145,  99,  90,  73, 131, 133, 130, 109,  90, 108, 116, 133, 131]])
    text_lens = torch.tensor([texts_len]).to(device)
    max_len = torch.from_numpy(np.array(texts_len)).to(device)
    pytorch_result = model(speakers.to(device), texts.to(device), text_lens.to(device), max_len.to(device))

    pytorch_result_ = pt_model(speakers.to(device), texts.to(device), text_lens.to(device), max_len.to(device))
    print('test')
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    # export_pt_to_onnx()
    # export_onnx(model)
    # xx1 = torch.randint(1, 10, (1, 10, 256)).to(device)
    # xx2 = torch.randint(1, 10, (1, 10)).to(device)
    # xx = LengthRegulator()
    # scripted_cell = torch.jit.script(LengthRegulator())
    # print(scripted_cell.code)
    # trace_model(model)
    # eval_onnx(model)
    # eval_pt(model)
    pt_model = torch.jit.load('./xx.pt')
    pt_model.eval()
    # message = evaluate(model, args.restore_step, configs)
    message = evaluate(pt_model, args.restore_step, configs)
    print(message)