import chainer
import chainer.functions as F
import numpy

from cppn import input_data, model


class MyUpdater(chainer.training.StandardUpdater):
    """ Update discriminator and generator with unconditional generation """
    def __init__(self, iterator: chainer.iterators.SerialIterator, gen: model.CPPN, dis: chainer.Chain,
                 gen_opt: chainer.optimizer.Optimizer, dis_opt: chainer.optimizer.Optimizer,
                 input_data: input_data.InputData, n_discriminator_update: int, device: int):
        self.generator = gen
        self.discriminator = dis
        self.gen_opt = gen_opt
        self.dis_opt = dis_opt
        self.input_data = input_data
        self.n_discriminator_update = n_discriminator_update

        optimizers = {'gen_opt': gen_opt, 'dis_opt': dis_opt}
        iterator = {'main': iterator}
        super(MyUpdater, self).__init__(iterator=iterator, optimizer=optimizers, device=device)

    def update_core(self):
        gen_opt = self.get_optimizer('gen_opt')
        dis_opt = self.get_optimizer('dis_opt')
        xp = self.generator.xp

        for i in range(self.n_discriminator_update):
            batch = self.get_iterator('main').next()
            batch_size = len(batch)
            x, z, c = [], [], []
            for b in batch:
                x.append(b[0])
                z.append(b[1])
                c.append(b[2])

            # ndarray -> variable
            x = chainer.Variable(xp.asarray(numpy.concatenate(x)))
            z = chainer.Variable(xp.asarray(numpy.concatenate(z)))
            c = chainer.Variable(xp.asarray(numpy.concatenate(c)))

            if i == 0:
                # generator
                x_fake = self.generator.forward(x, z)
                y_fake = self.discriminator.forward(x_fake)
                loss_gen = F.sum(F.softplus(-y_fake)) / batch_size
                self.generator.cleargrads()
                loss_gen.backward()
                gen_opt.update()
                chainer.reporter.report({'loss_gen': loss_gen})

            y_real = self.discriminator.forward(c)

            z = self.input_data.sample_z(batch_size)
            z = chainer.Variable(xp.asarray(z))

            x_fake = self.generator(x, z)
            y_fake = self.discriminator(x_fake)
            x_fake.unchain_backward()

            loss_dis = F.sum(F.softplus(-y_real)) / batch_size
            loss_dis += F.sum(F.softplus(y_fake)) / batch_size

            self.discriminator.cleargrads()
            loss_dis.backward()
            dis_opt.update()

            chainer.reporter.report({'loss_dis': loss_dis})
