class EvenOdd:
    def __init__(self):
        # Theorem grammar.
        self.given = [
            "Given ",
            "For ",
            "Assuming ",
            "Taking ",
            "Using ",
            "With ",
        ]
        self.any = [
            "any ",
            "all ",
            "some ",
            "every ",
            ""
        ]
        self.natural = [
            "natural numbers ",
            "positive integers ",
            "integers ",
            "positive numbers ",
            "whole numbers ",
            "natural terms ",
            "positive terms ",
            r"\in \mathbb{N} ",
            r"\in \mathbb{Z}^+ ",
            "unknowns ",
            "variables ",
            "terms ",
            ""
        ]
        self.expression = [
            "the sum ",
            "the expression ",
            "the formula ",
            ", "
        ]
        self.must_be = [
            "is ",
            "must be ",
            "will be ",
            "is guaranteed to be "
        ]
        self.times = [
            "* ",
            r"\cdot ",
            r"\times ",
            ""
        ]
        self.notice = [
            "notice ",
            "recall ",
            "observe ",
            "take note ",
            "see ",
            "remember ",
            "we know "
        ]
        self.that = [
            "that ",
            ""
        ]
        self.is_ = [
            "is ",
            "is known to be ",
            "is clearly ",
            "is obviously ",
            "is trivially ",
            "is known as ",
            "is guaranteed to be "
        ]
        self.an = [
            "an "
        ]
        self.number = [
            "number ",
            "natural ",
            "natural number ",
            "integer ",
            "positive integer ",
            "whole number ",
            "number in $ \mathbb{N} $ ",
            "number in $ \mathbb{Z}^+ $ ",
            "element of $ \mathbb{N} $ ",
            "element of $ \mathbb{Z}^+ $ "
        ]
        self.since = [
            "since ",
            "because ",
            "as a consequence that ",
            "from the fact ",
            "from the knowledge ",
            "by reason that "
        ]
        self.addition = [
            "the addition ",
            "the summation ",
            "the summing ",
            "the adding ",
            "the totaling "
        ]
        self.of = [
            "of ",
            "between "
        ]
        self.numbers = [
            "numbers ",
            "naturals ",
            "natural numbers ",
            "integers ",
            "positive integers ",
            "whole numbers ",
            "numbers in $ \mathbb{N} $ ",
            "numbers in $ \mathbb{Z}^+ $ ",
            "elements of $ \mathbb{N} $ ",
            "elements of $ \mathbb{Z}^+ $ "
        ]
        self.together = [
            "together ",
            "collectively ",
            "successively ",
            "concurrently ",
            ""
        ]
        self.with_ = [
            "with ",
            "and "
        ]
        self.itself = [
            "itself ",
            "in itself ",
            "by itself ",
            "by definition ",
            "fundamentally ",
            "by its very nature ",
            "instrinsically ",
            "clearly ",
            "obviously ",
            "trivially ",
            "", "", "", "", "", "", "", "", "", "", "", ""
        ]
        self.and_ = [
            "and ",
            "and likewise ",
            "in addition to ",
            "plus ",
            "and also ",
            "additionally ",
            "as well as ",
            "and moreover "
        ]
        self.multiplication = [
            "the multiplication ",
            "the product ",
            "multiplying ",
            "taking the product "
        ]
        self.a = [
            "a ",
            "some ",
            "any ",
            "an arbitrary "
        ]
        self.then = [
            "then ",
            "as a consequence ",
            "as a result ",
            "hence ",
            "thus ",
            ""
        ]
        self.expression = [
            "the expression ",
            "the formula ",
            "the sum ",
            "the summation ",
            "", ""
        ]
        self.our = [
            "our ",
            "the ",
            "this "
        ]
        self.theorem = [
            "theorem ",
            "result ",
            "main result ",
            "lemma ",
            "proposition ",
            "claim "
        ]
        self.holds = [
            "holds ",
            "is true ",
            "must be true ",
            "is proven "
        ]
        self.further = [
            "further ",
            "furthermore ",
            "additionally ",
            "likewise ",
            "in addition ",
            "adding on ",
            "building on ",
            "what's more ",
            "on top of this ",
            "in addition to this "
        ]
        self.therefore = [
            "therefore , ",
            "thus , ",
            "hence , ",
            "as a consequence , ",
            "accordingly , ",
            "so , ",
            "then , ",
            "consequently , ",
            "for this reason , ",
            "in consequence , "
        ]
        self.show = [
            "we show ",
            "we demonstrate ",
            "we prove ",
            "we justify ",
            "we verify ",
            "we check ",
            "it can be shown that ",
            "it can be demonstrated that ",
            "it can be proven that ",
            "it can be justified that ",
            "it can be verified that ",
            "it can be checked that ",
        ]
        self.using = [
            "using ",
            "with ",
            "applying ",
            "in accordance with ",
            "according to ",
            "in connection with ",
            "employing ",
            "making use of ",
            "by using ",
            "by applying ",
            "by employing "
        ]
        self.fact = [
            "the fact ",
            "the knowledge ",
            "the understanding ",
            ""
        ]
        self.first = [
            "first , ",
            "starting , ",
            "to start , ",
            "to begin , ",
            "to begin with , ",
            "to start with , "
        ]
        self.next = [
            "next , ",
            "additionally , ",
            "following , ",
            "likewise , ",
            "in addition , ",
            "subsequently , ",
            "next off , ",
            "similarly , "
        ]
        self.use = [
            "we use ",
            "we apply ",
            "we employ ",
            "we make use ",
            "we utilize ",
            "we work with the assumption ",
            "we use the assumption ",
            "we apply the assumption "
        ]
        self.also = [
            "also ",
            "as well ",
            "additionally ",
            "in addition ",
            "too ",
            "at the same time "
        ]
        self.coefficient = [
            "the coefficient ",
            "the number ",
            "the natural number ",
            "the whole number ",
            "the integer ",
            "the positive integer ",
            "the leading term "
        ]
        self.product = [
            "the product ",
            "the pair ",
            "the term ",
            ""
        ]
        self.this = [
            "this ",
            "this fact ",
            "this claim "
        ]
        self.which = [
            "which ",
            "and "
        ]
        self.coefficients = [
            "the coefficients ",
            "the numbers ",
            "the natural numbers ",
            "the whole numbers ",
            "the integers ",
            "the positive integers ",
            "the leading terms "
        ]
        self.products = [
            "the products ",
            "the pairs ",
            "the terms "
        ]
        self.are = [
            "are ",
            "must be ",
            "will be ",
            "are known to be ",
            "are clearly ",
            "are obviously ",
            "are trivially ",
            "are known as ",
            "are guaranteed to be "
        ]
        self.themselves = [
            "themselves ",
            "in themselves ",
            "by themselves ",
            "by definition ",
            "fundamentally ",
            "by their very nature ",
            "instrinsically ",
            "clearly ",
            "obviously ",
            "trivially ",
            "", "", "", "", "", "", "", "", "", "", "", ""
        ]
        self.trivially = [
            "trivially ",
            "clearly ",
            "obivously ",
            ""
        ]
        
class Composites():
    def __init__(self):
        self.equal = [
            "= ",
            "$ is equal to $ ",
            "$ is equivalent to $ ",
            "$ is the same as $ ",
            "$ equals $ ",
            "$ is $ "
        ]
        self.given = [
            "given ",
            "for ",
            "assuming ",
            "taking ",
            "using ",
            "with "
        ]
        self.greater = [
            "greater than ",
            "larger than ",
            "more than "
        ]
        self.therefore = [
            "therefore , ",
            "thus , ",
            "as a consequence , ",
            "hence , ",
            "so , ",
            "because of this , ",
            "consequently , ",
            "as a result , "
        ]
        self.let = [
            "let ",
            "take ",
            "assume ",
            "allow "
        ]
        self.when = [
            "if ",
            "when ",
            "given ",
            "assuming "
        ]
        self.such_that = [
            "such that ",
            "so that ",
            "where ",
            "satisfying the condition that "
        ]
        self.number = [
            "number ",
            "integer ",
            "natural number ",
            "positive integer ",
            "whole number ",
            ""
        ]
        self.times = [
            "* ",
            r"\cdot ",
            r"\times "
        ]
        self.said = [
            "is ",
            "is said to be ",
            "is defined to be ",
            "is defined as ",
            "will be ",
            "is called ",
            "is named ",
            "is considered ",
            "is considered to be ",
            "will be called ",
            "will be considered ",
            "will be named ",
            "will be considered as ",
            "will be considered to be "
        ]
        self.numbers = [
            "numbers ",
            "integers ",
            "natural numbers ",
            "positive integers ",
            "whole numbers ",
            ""
        ]
        self.listing = [
            "$ and $ ",
            "$ , and $ ",
            "$ , $ "
        ]
        self.exist = [
            "there are $ ",
            "there exist $ ",
            "$ \exists "
        ]
        self._if = [
            "if ",
            "when ",
            "given ",
            "if and only if ",
            "only if ",
            "iff ",
            "whenever ",
            "wherever ",
            "assuming that ",
            "given that ",
            "under the condition that ",
            "supposing that ",
            "supposing ",
            "granted that ",
            "granted ",
            "assuming ",
            "under the condition ",
            "with the condition ",
            "with the condition that "
        ]
        self._is = [
            "is "
        ]
        self.some = [
            "some "
        ]
        self.have = [
            "we have ",
            "we get "
        ]
        self._for = [
            "for ",
            "given ",
            "assuming ",
            "with "
        ]
        self.be = [
            "be ",
            "as ",
            "are "
        ]
        self.one = [
            "<nat:1> "
        ]
        self.two = [
            "<nat:2> "
        ]
        self._and = [
            "and ",
            "as well as ",
            "plus "
        ]
        self.then = [
            "then ",
            ""
        ]
        self.factors = [
            "factors ",
            "terms ",
            "unique factors ",
            "unique terms "
        ]
        self.equal_to = [
            "or equal to ",
            "or equivalent to "
        ]
        self.so = [
            "so ",
            "thus ",
            "hence ",
            "accordingly ",
            "then ",
            "therefore ",
            "consequently ",
            "for this reason "
        ]
        self.product = [
            "the product ",
            "the multiplication "
        ]
        self.of = [
            "of ",
            "between "
        ]
        self.note = [
            "note that ",
            "note ",
            "notice that ",
            "notice ",
            "observe that ",
            "observe "
        ]
        self.recall = [
            "recall that ",
            "remember that ",
            "recollect that ",
            "recall ",
            "remember "
        ]
        self.this = [
            "this "
        ]
        self.gives = [
            "implies ",
            "suggests ",
            "gives ",
            "yields ",
            "hints ",
            "shows ",
            "produces ",
            "indicates "
        ]
        self.proven = [
            "is proven ",
            "is validated ",
            "is verified ",
            "is true "
            "has been proven ",
            "has been validated ",
            "has been verified "
        ]
        self.further = [
            "further ",
            "also ",
            "in addition ",
            "moreover ",
            "what's more ",
            "furthermore ",
            "additionally ",
            "plus ",
            "similarly "
        ]
        self.prove = [
            "we show ",
            "we demonstrate ",
            "we prove ",
            "we present ",
            "we verify ",
            "we confirm ",
            "we justify "
        ]
        self.show = self.prove
        self.our = [
            "our ",
            "the "
        ]
        self.also = [
            "also ",
            "further ",
            "likewise ",
            "still ",
            "too ",
            "additionally ",
            "as well ",
            "moreover "
        ]
        self.holds = [
            "holds ",
            "is true ",
            "is valid "
        ]
        self.since = [
            "because ",
            "as ",
            "by reason of ",
            "considering ",
            "for ",
            "in consideration of ",
            "in view of ",
            "on account of ",
            "seeing that "
        ]
        self.theorem = [
            "theorem ",
            "result ",
            "belief ",
            "statement ",
            "principle ",
            "proposition ",
            "formula ",
            "theory ",
            "thesis ",
            "claim "
        ]
        self.are = [
            "are "
        ]
        self.say = [
            "we denote that ",
            "denote that ",
            "define that ",
            "we define that ",
            "we say that ",
            "say that ",
            "we assume that ",
            "assume that ",
            "take that ",
            "take as given that ",
            "let that ",
            "we claim that ",
            "claim that "
        ]

class Powers(Composites):
    def __init__(self):
        super().__init__()
        self.exponent = [
            "the exponent ",
            "the exponential ",
            "the repeated product ",
            "the repeated multiplication ",
            "the raising of "
        ]
        self.raised = [
            "taken to the ",
            "raised to the ",
            "exponentiated to the ",
            "repeatedly multiplied to the ",
        ]
        self.power = [
            "power ",
            "exponent ",
            "exponential ",
            "degree "
        ]
        self.exists = [
            "there is $ ",
            "there exists $ ",
            "$ \exists "
        ]
        self.base = [
            "base "
        ]

class Poly():
    def __init__(self):
        self.consider = [
            "Consider the following ",
            "Take the following ",
            "Let the following ",
            "We define the following ",
            "Define the following ",
            "Consider a ",
            "Take a ",
            "We define a ",
            "Define a "
        ]
        self.program = [
            "program ",
            "segment of code ",
            "set of instructions ",
            "set of commands ",
            "list of instructions ",
            "list of commands ",
            "series of instructions ",
            "series of commands ",
            "code segment ",
            "code "
        ]
        self.such = [
            "as follows ",
            "such that ",
            "as ",
            ": "
        ]
        self.assume = [
            "Given ",
            "Assuming ",
            "Assume ",
            "Allowing ",
            "Allow "
        ]
        self.forall = [
            "for any ",
            "for every ",
            "for all ",
            "given any ",
            "given every ",
            "given all ",
            "letting ",
            "assuming some ",
            "taking ",
            "with some ",
            "given some ",
            "for some ",
            "for ",
            "given "
        ]
        self.natural = [
            "natural numbers ",
            "positive integers ",
            "integers ",
            "positive numbers ",
            "whole numbers ",
            "natural coefficients ",
            "positive coefficients ",
            "positive integer coefficients ",
            r"\in \mathbb{N} ",
            r"\in \mathbb{Z}^+ ",
            ""
        ]
        self.that = [
            "that ",
            "that we have ",
            "it to be that ",
            "it to hold that ",
            ""
        ]
        self.prior = [
            "before ",
            "prior to ",
            "ahead of "
        ]
        self.execution = [
            "the execution of ",
            "running ",
            "executing ",
            "interpreting ",
            "evaluating ",
            "the evaluation of "
        ]
        self.this = [
            "this ",
            "our ",
            "the "
        ]
        self.then = [
            "then ",
            "it follows that ",
            "it holds that ",
            "it must be that "
        ]
        self.have = [
            "we have ",
            "we see ",
            ""
        ]
        self.times = [
            "* ",
            r"\cdot ",
            r"\times "
        ]
        self.after = [
            "after ",
            "once ",
            "when "
        ]
        self.done = [
            "has finished ",
            "is done ",
            "has executed ",
            "has finished executing ",
            "is done executing ",
            "has terminated ",
            "has exited ",
            "exits ",
            "terminates ",
            "executes ",
            "finishes "
        ]
        self.the_proof = [
            "The proof ",
            "Correctness for ",
            "The proof of correctness for ",
            "The correctness of ",
            "Validating ",
            "Proving the correctness of ",
            "Demonstrating the correctness of "
        ]
        self.demonstrated = [
            "is done ",
            "is accomplished ",
            "is shown ",
            "is given ",
            "is demonstrated ",
            "can be seen ",
            "is evident ",
            "is clear ",
            "can be demonstrated ",
            "can be shown ",
            "can be given ",
            "can be accomplished ",
            "can be done "
        ]
        self.by = [
            "with ",
            "by ",
            "through ",
            "using "
        ]
        self.following = [
            "the following ",
            "the below ",
            "this "
        ]
        self.observe = [
            "Note that ",
            "Observe that ",
            "Observe ",
            "Notice ",
            "See that ",
            "We can see that ",
            "Let "
        ]
        self.applying = [
            "Applying ",
            "Using ",
            "Utilizing ",
            "The application of ",
        ]
        self.hoare = [
            "standard Hoare ",
            "typical Hoare ",
            "usual Hoare ",
            "Hoare "
        ]
        self.logic = [
            "logic ",
            "reasoning ",
            "techniques "
        ]
        self.gives = [
            "gives ",
            "yields ",
            "results in ",
            "gives us ",
            "returns ",
            "produces "
        ]
        self.further = [
            "Furthering ",
            "Continuing ",
            "Allowing ",
            "Advancing "
        ]
        self.therefore = [
            "Thus , ",
            "Hence , ",
            "Therefore , ",
            "As a consequence , ",
            "As a result , ",
            "Considering the above , ",
            "Given the preceding argument , ",
            "Considering the preceding argument , ",
            "Given the above , ",
            "Thusly , "
        ]
        self.proven = [
            "has been proven ",
            "has been shown ",
            "has been demonstrated to be ",
            "has been shown to be ",
            "is proven ",
            "is shown ",
            "is demonstrated to be ",
            "is validated as "
        ]
        self.completes = [
            "resolves the ",
            "solves the ",
            "finishes the ",
            "completes the ",
            "accomplishes the ",
            "finalizes the ",
            "ends the "
        ]
        self.proof = [
            "proof ",
            "goal ",
            "argument "
        ]
        self.the_proof = [
            "The proof of ",
            "The argument for ",
            "The course of logic for ",
            "The flow of logic for ",
            "The method of proof for ",
            "The way to prove correctness for ",
            "Proving correctness for ",
            "Arguing the correctness of ",
            "Showing our claim about correctness for ",
            "Demonstrating correctness of "
        ]
        self.considering = [
            "considering ",
            "looking at ",
            "taking a look at ",
            "shifting focus toward ",
            "focusing on ",
            "paying attention to ",
            "moving on to ",
            "advancing to ",
            "moving forward to ",
            "taking another step toward ",
            "honing in on ",
            "jumping to ",
            "observing "
        ]
        self.the_next = [
            "the next ",
            "the following ",
            "the upcoming ",
            "the subsequent ",
            "the proceeding "
        ]
        self.step = [
            "step ",
            "command ",
            "statement ",
            "line ",
            "piece of code ",
            "line of code ",
            "bit of code "
        ]
        self.steps = [
            "steps ",
            "commands ",
            "statements ",
            "lines ",
            "pieces of code ",
            "lines of code",
            "bits of code"
        ]
        self.in_the = [
            "in the ",
            "contained in the ",
            "in our ",
            "contained in our ",
            "found in the ",
            "found in our ",
            "seen in the ",
            "seen in our "
        ]
        self.next = [
            "next ",
            "following ",
            "immediately following ",
            "after ",
            "further ",
            "immediately after ",
            "now ",
            "moving on ",
            "stepping forward ",
            "moving forward ",
            "advancing ",
            "advancing on ",
            "stepping through ",
            "keeping momentum ",
            "moving on to the next step ",
            "advancing to the next step ",
            "going forward ",
            "moving forward to the next step ",
            "moving onward ",
            "going to the next step ",
        ]
        self.given = [
            "this is given ",
            "this is guaranteed ",
            "this is ensured ",
            "this is shown to be logically sound ",
            "this is sound ",
            "this is known to be sound ",
            "this is rigorously given ",
            "this is shown ",
            "this is deduced "
        ]
        self.rule = [
            "rule ",
            "inference rule ",
            "scheme ",
            "law "
        ]
        self.of = [
            "of ",
            "defined by ",
            "given in ",
            "defined in ",
            "given by ",
            "found in ",
            "obtained from ",
            "in ",
            "from "
        ]
        self.assignment = [
            "assignment ",
            "binding ",
            "variable binding ",
            "variable assignmet ",
            "environment update ",
            "variable update ",
            ""
        ]
        self.command = [
            "command ",
            "statement ",
            "line ",
            "code "
        ]
        self.assigning = [
            "assigning ",
            "binding ",
            "letting ",
            "giving ",
            "yielding ",
            "defining "
        ]
        self.takes = [
            "takes ",
            "moves ",
            "transfers ",
            "advances ",
            "forwards ",
            "redefines "
        ]
        self.precondition = [
            "the precondition ",
            "the condition ",
            ""
        ]
        self.to = [
            "to ",
            "toward ",
            "as ",
            "into "
        ]
        self.postcondition = [
            "the postcondition ",
            "the condition ",
            ""
        ]