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

//Admin
//On second though, admin is just going to hold *all* the neural networks, and maybe
//categorize them, or link them in some kind of graph. We'll need this for training,
//so that if we're learning organically (aka through some sort of edge interface like
//a camera or microphone, and receiving aural or visual feedback), we can trace the
//feedback to a desired outcome and use that information for back-propogation
type Admin struct {
	Networks   []*Network          //I'll want this better organized later, as certain systems may end up auto-generating thousands of thousands of networks
	NetworkMap map[string]*Network //I'll have to implement some sort of string-matching algorithm here
}
