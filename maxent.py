import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import MaxentClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import movie_reviews

def word_feats(words):
	return dict([(word, True) for word in words])

def word_feats_stopwords(words):
	swords = stopwords.words('english')
	return dict([(word, True) for word in words if word not in swords])

def word_feats_punctuations(words):
	sentence = ' '.join(words)
	tokenizer = RegexpTokenizer(r'\w+')
	words = tokenizer.tokenize(sentence)
	return dict([(word, True) for word in words])

def word_feats_lemmatize(Lwords):
	words = []
	for each in Lwords:
		lmtzr = WordNetLemmatizer()
		words.append(lmtzr.lemmatize(each))

	return dict([(word, True) for word in words])

def classification(negfeats, posfeats, pospercent, negpercent):
	negWords = [each[0] for each in negfeats]
	posWords = [each[0] for each in posfeats]

	negcutoff = int(len(negfeats)*negpercent)
	poscutoff = int(len(posfeats)*pospercent)


	trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]

	#classifier = nltk.MaxentClassifier.train(trainfeats)

	algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
	classifier = nltk.MaxentClassifier.train(trainfeats, algorithm, max_iter=2)

	#classifier.show_most_informative_features(100)

	all_words = nltk.FreqDist(word for word in movie_reviews.words())
	top_words = set(all_words.keys()[:])

	return classifier



def calculateScore(classifier, to_review_words):
	neg = 0
	pos = 0
	for each in to_review_words:
		if classifier.classify(word_feats(each)) == 'neg':
			neg += 1

		if classifier.classify(word_feats(each)) == 'pos':
			pos += 1

	print neg,pos, pos-neg
	return neg,pos

def calculateScore_stopwords(classifier, to_review_words):
	neg = 0
	pos = 0
	for each in to_review_words:
		#print classifier.classify(word_feats_stopwords(each))
		if classifier.classify(word_feats_stopwords(each)) == 'neg':
			neg += 1
			#print each,'this is neg'
		if classifier.classify(word_feats_stopwords(each)) == 'pos':
			pos += 1
			#print each,'this is pos'
	print neg,pos, pos-neg
	return neg,pos

def calculateScore_punctuations(classifier, to_review_words):
	neg = 0
	pos = 0
	for each in to_review_words:
		if classifier.classify(word_feats_punctuations(each)) == 'neg':
			neg += 1
		if classifier.classify(word_feats_punctuations(each)) == 'pos':
			pos += 1

	print neg,pos, pos-neg
	return neg,pos


def calculateScore_lemmatizer(classifier, to_review_words):
	neg = 0
	pos = 0
	for each in to_review_words:
		if classifier.classify(word_feats_lemmatize(each)) == 'neg':
			neg += 1
		if classifier.classify(word_feats_lemmatize(each)) == 'pos':
			pos += 1

	print neg,pos, pos-neg
	return neg,pos



def main():
	negids = movie_reviews.fileids('neg')
	posids = movie_reviews.fileids('pos')

	to_review1 = "A man with a magnanimous spirit helps a mute girl from Pakistan return home."
	to_review2 = "Forced out of his own company by former Darren Cross, Dr. Hank Pym (Michael Douglas) recruits the talents of Scott Lang (Paul Rudd), a master thief just released from prison. Lang becomes Ant-Man, trained by Pym and armed with a suit that allows him to shrink in size, possess superhuman strength and control an army of ants. The miniature hero must use his new skills to prevent Cross, also known as Yellowjacket, from perfecting the same technology and using it as a weapon for evil."
	to_review3 = '''Parents need to know that kids may clamor to see this fast-paced, action-packed comic book-based adventure. But it's definitely more age-appropriate for teens than younger children. Although much of the violence is clearly meant to be based in the realm of sci-fi and fantasy -- and/or is shown at a distance -- there's plenty of it, from massive explosions to children held at gunpoint to super-powered fistfights. Some of the violence is war themed, and some characters get hurt and/or die. While much is made of lead character Tony Stark's devil-may-care lifestyle of fun and frolic, viewers also see him turn away from the more irresponsible aspects of playboyhood. Language is minimal, and sexual content is more suggested than shown overall -- though there are a few eyebrow-raising moments.'''
	reviews = []
	reviews.append(to_review1)
	reviews.append(to_review2)
	reviews.append(to_review3)

	for to_review in reviews:
		to_review_words = to_review.split(" ")
		print "Reviewing",to_review,"\n\n\n"


		print ''' Normal classification ''',"\n\n"
		negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
		posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
		calculateScore(classification(negfeats, posfeats, 1, 1), to_review_words)
		calculateScore(classification(negfeats, posfeats, 1, 0.95), to_review_words)
		calculateScore(classification(negfeats, posfeats, 0.95, 1), to_review_words)
		calculateScore(classification(negfeats, posfeats, 0.9, 1), to_review_words)
		calculateScore(classification(negfeats, posfeats, 1, 0.9), to_review_words)

		print ''' Without Punctuations ''',"\n\n"
		negfeats_stopwords = [(word_feats_punctuations(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
		posfeats_stopwords = [(word_feats_punctuations(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
		calculateScore_punctuations(classification(negfeats, posfeats, 1, 1), to_review_words)
		calculateScore_punctuations(classification(negfeats, posfeats, 1, 0.95), to_review_words)
		calculateScore_punctuations(classification(negfeats, posfeats, 0.95, 1), to_review_words)
		calculateScore_punctuations(classification(negfeats, posfeats, 0.9, 1), to_review_words)
		calculateScore_punctuations(classification(negfeats, posfeats, 1, 0.9), to_review_words)



		print ''' Without Stop Words ''',"\n\n"
		negfeats_stopwords = [(word_feats_stopwords(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
		posfeats_stopwords = [(word_feats_stopwords(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
		wordstoreview = []
		for each in to_review_words:
			if each not in stopwords.words('english'):
				wordstoreview.append(each)
		calculateScore_stopwords(classification(negfeats, posfeats, 1, 1), wordstoreview)
		calculateScore_stopwords(classification(negfeats, posfeats, 1, 0.95), to_review_words)
		calculateScore_stopwords(classification(negfeats, posfeats, 0.95, 1), to_review_words)
		calculateScore_stopwords(classification(negfeats, posfeats, 0.9, 1), to_review_words)
		calculateScore_stopwords(classification(negfeats, posfeats, 1, 0.9), to_review_words)


		print ''' With Lemmatizer ''',"\n\n"
		negfeats_stopwords = [(word_feats_lemmatize(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
		posfeats_stopwords = [(word_feats_lemmatize(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
		calculateScore_lemmatizer(classification(negfeats, posfeats, 1, 1), to_review_words)
		calculateScore_lemmatizer(classification(negfeats, posfeats, 1, 0.95), to_review_words)
		calculateScore_lemmatizer(classification(negfeats, posfeats, 0.95, 1), to_review_words)
		calculateScore_lemmatizer(classification(negfeats, posfeats, 0.9, 1), to_review_words)
		calculateScore_lemmatizer(classification(negfeats, posfeats, 1, 0.9), to_review_words)



if __name__ == '__main__':
	main()