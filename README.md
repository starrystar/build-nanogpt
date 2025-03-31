# build nanoGPT

这个仓库包含了对 nanoGPT (https://github.com/karpathy/nanoGPT/tree/master) 的从零开始的复现。Git 提交记录是精心保留的。

我们基本上是从一个空文件开始，逐步实现 GPT-2 （124M 参数）模型的复现。如果你有更多耐心或预算，这套代码也可以用于训练更大的 [GPT-3](https://arxiv.org/pdf/2005.14165) 模型。虽然 GPT-2（124M）模型早在 2019 年（大约五年前）训练时花费了相当长的时间，但在今天，复现它只需约 1 小时和约 10 美元的成本。如果你本地没有足够的 GPU 资源，建议使用云服务上的 GPU 机器，我推荐使用 Lambda Labs 。

请注意，GPT-2 和 GPT-3 都是简单的语言模型，它们是在互联网文本上训练的，其功能本质上就是“生成”互联网风格的文本。因此，本仓库/视频并未涉及像 ChatGPT 那样的对话式微调（Chat fine-tuning），你也不能像与 ChatGPT 对话那样与它互动。微调过程（尽管从概念上讲很简单——SFT 只需更换数据集并继续训练）属于后续步骤，我们将在以后的内容中再做讲解。目前来看，当你用 "Hello, I'm a language model" 这样的提示词去引导这个经过约 100 亿 token 训练的 124M 模型时，它会输出类似下面这样的内容：

```
Hello, I'm a language model, and my goal is to make English as easy and fun as possible for everyone, and to find out the different grammar rules
Hello, I'm a language model, so the next time I go, I'll just say, I like this stuff.
Hello, I'm a language model, and the question is, what should I do if I want to be a teacher?
Hello, I'm a language model, and I'm an English person. In languages, "speak" is really speaking. Because for most people, there's
```

And after 40B tokens of training:

```
Hello, I'm a language model, a model of computer science, and it's a way (in mathematics) to program computer programs to do things like write
Hello, I'm a language model, not a human. This means that I believe in my language model, as I have no experience with it yet.
Hello, I'm a language model, but I'm talking about data. You've got to create an array of data: you've got to create that.
Hello, I'm a language model, and all of this is about modeling and learning Python. I'm very good in syntax, however I struggle with Python due
```

For discussions and questions, please use [Discussions tab](https://github.com/karpathy/build-nanogpt/discussions), and for faster communication, have a look at my [Zero To Hero Discord](https://discord.gg/3zy8kqD9Cp), channel **#nanoGPT**:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## Video

[Let's reproduce GPT-2 (124M) YouTube lecture](https://youtu.be/l8pRSuU81PU)

## (Green) Errata

小的清理工作：在我们切换到使用 flash attention 后，忘记删除之前用于注册 bias 的 register_buffer，这个问题最近通过一个 PR 得到了修复。

torch.autocast 函数需要一个参数 device_type，我之前固执地直接传入了完整的 device（比如 cuda:3），希望它也能正常工作。但实际上，在某些版本的 PyTorch 中，它确实要求只传设备类型，这会导致错误。所以我们需要把像 cuda:3 这样的设备名称简化为 cuda。目前，对于 Apple Silicon 上的 mps 设备，它会被当作 cpu 类型处理，我不太确定这是否是 PyTorch 的预期行为。

令人困惑的是，model.require_backward_grad_sync 实际上会在前向传播和反向传播中都被用到。因此我们把这个语句提前，以确保它也作用于前向传播过程。

## Prod

For more production-grade runs that are very similar to nanoGPT, I recommend looking at the following repos:

- [litGPT](https://github.com/Lightning-AI/litgpt)
- [TinyLlama](https://github.com/jzhang38/TinyLlama)

## FAQ

## License

MIT
