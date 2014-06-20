import numpy as np
from pylearn2.utils import safe_izip
from pylearn2.utils.iteration import SubsetIterator
from pylearn2.utils.iteration import FiniteDatasetIterator


class SequenceDatasetIterator(FiniteDatasetIterator):

    def get_seq(self, ind):
        """
        return seq_len words before ind, including ind
        while paying attention to sentence structure and filling
        beginning and end of setences with special character
        """

        return format_sentence(data = self._raw_data[0],
                        seq_len = self._dataset.seq_len,
                        ind = ind - 1,
                        begin = self._dataset.begin_sentence,
                        end = self._dataset.end_sentence)

    def next(self):

        next_index = self._subset_iterator.next()
        targets = False
        aux_targets = False

        y = self._raw_data[0][next_index].reshape((self.batch_size, 1))

        if isinstance(next_index, slice):
            next_index = slice_to_list(next_index)

        x = np.zeros((self.batch_size, self._dataset.seq_len))
        x = np.asarray([self.get_seq(i) for i in xrange(self.batch_size)])

        y = self._dataset.mapped_dict[y]

        rval = (self._convert[0](x), self._convert[1](y))

        return rval


def slice_to_list(item):
    ifnone = lambda a, b: b if a is None else a
    return list(range(ifnone(item.start, 0), item.stop, ifnone(item.step, 1)))


def format_sentence(data, ind, seq_len, begin, end):
    """

    Parameters
    ----------
    begin: int
        index of the start of sentence <S>
    end: index of end of sentence </S>
    """

    rval = np.ones((seq_len)) * end
    if ind > seq_len:
        rval[:] = data[ind-seq_len:ind].flatten()
    elif ind > 0:
        rval[seq_len-ind:] =  data[:ind].flatten()

    w = np.where(rval == -1)[0]
    if len(w) > 0:
        rval[0:max(0, w[-1])] = end
        rval[w[-1]] = begin

    return rval


class NoiseIterator(FiniteDatasetIterator):

    def __init__(self, dataset, subset_iterator, data_specs=None,
            return_tuple=False, convert=None, noise_p=None, num_noise=None):

        super(NoiseIterator, self).__init__(dataset=dataset,
                                            subset_iterator=subset_iterator,
                                            data_specs=data_specs,
                                            return_tuple=return_tuple,
                                            convert=convert)
        self.noise_p = noise_p
        self.num_noise = num_noise


    def get_noise(self):
        rng = self._dataset.rng
        rval = rng.multinomial(n=1, pvals=self.noise_p, size=(self.batch_size * self.num_noise))
        return np.argmax(rval, axis=1)

    def next(self):
        """
        Retrieves the next batch of examples.

        Returns
        -------
        next_batch : object
            An object representing a mini-batch of data, conforming
            to the space specified in the `data_specs` constructor
            argument to this iterator. Will be a tuple if more
            than one data source was specified or if the constructor
            parameter `return_tuple` was `True`.

        Raises
        ------
        StopIteration
            When there are no more batches to return.
        """
        next_index = self._subset_iterator.next()
        # TODO: handle fancy-index copies by allocating a buffer and
        # using np.take()


        rval = list(
            fn(data[next_index]) if fn else data[next_index]
            for data, fn in safe_izip(self._raw_data, self._convert))
        if len(self._source) > 1:
            rval.append(self.get_noise())
        rval = tuple(rval)
        if not self._return_tuple and len(rval) == 1:
            rval, = rval
        return rval

