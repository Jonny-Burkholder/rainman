package neuralnetwork

/*Admin allows us to connect a parent NN with a series of child NN. For instance,
We may have an input NN that accepts images as input, and decides if the image is
A face, an animal, a machine, or an object. From there, it will pass the image to the
appropriate child NN - for instance, if the parent network decides that it is looking
at a face, it will pass the image to a child NN that specializes in facial recognition,
so that the system may then determine *which* face it is looking at. This in turn frees
the parent NN to accept new input. This will most likely be a tree or graph structure*/

//I've decided that this is probably better done on the network level. Thus this is
//deprecated before it begins, but I want to keep the above comment for reference
