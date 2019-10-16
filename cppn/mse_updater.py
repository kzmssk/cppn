import chainer
import chainer.functions as F
import numpy

from cppn import input_data, model


class MseUpdater(chainer.training.StandardUpdater):
    """ Update discriminator and generator with unconditional generation """
    def __init__(self, iterator: chainer.iterators.SerialIterator, gen: model.CPPN,
                 gen_opt: chainer.optimizer.Optimizer, input_data: input_data.InputData, device: int):
        self.generator = gen
        self.gen_opt = gen_opt
        self.input_data = input_data

        optimizers = {'gen_opt': gen_opt}
        iterator = {'main': iterator}
        super(MseUpdater, self).__init__(iterator=iterator, optimizer=optimizers, device=device)

    def update_core(self):
        gen_opt = self.get_optimizer('gen_opt')
        xp = self.generator.xp

        batch = self.get_iterator('main').next()
        batch_size = len(batch)
        x, z, t = [], [], []
        for b in batch:
            x.append(b[0])
            z.append(b[1])
            t.append(b[2])

        # ndarray -> variable
        x = chainer.Variable(xp.asarray(numpy.concatenate(x)))
        z = chainer.Variable(xp.asarray(numpy.concatenate(z)))
        t = chainer.Variable(xp.asarray(numpy.concatenate(t)))

        # forward prop
        y = self.generator.forward(x, z)

        # loss
        loss = F.mean_squared_error(y, t) / batch_size
        
        # backward
        self.generator.cleargrads()
        loss.backward()

        # update
        gen_opt.update()

        # report log
        chainer.reporter.report({'loss_gen': loss})
