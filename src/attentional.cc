#include <queue>
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/lstm.h"

#include "kbestlist.h"
#include "bitext.h"
#include "attentional.h"

using namespace std;
using namespace cnn;

AttentionalModel::AttentionalModel(Model& model, unsigned src_vocab_size, unsigned tgt_vocab_size) {
  forward_builder = LSTMBuilder(lstm_layer_count, embedding_dim, half_annotation_dim, &model);
  reverse_builder = LSTMBuilder(lstm_layer_count, embedding_dim, half_annotation_dim, &model);
  output_builder = LSTMBuilder(lstm_layer_count, 2 * half_annotation_dim, output_state_dim, &model);
  p_Es = model.add_lookup_parameters(src_vocab_size, {embedding_dim});
  p_Et = model.add_lookup_parameters(tgt_vocab_size, {embedding_dim});
  p_aIH = model.add_parameters({alignment_hidden_dim, output_state_dim + 2 * half_annotation_dim});
  p_aHb = model.add_parameters({alignment_hidden_dim, 1});
  p_aHO = model.add_parameters({1, alignment_hidden_dim});
  p_aOb = model.add_parameters({1, 1});
  // The paper says s_0 = tanh(Ws * h1_reverse), and that Ws is an N x N matrix, but somehow below implies Ws is 2N x N.
  p_Ws = model.add_parameters({2 * half_annotation_dim, half_annotation_dim});
  p_bs = model.add_parameters({2 * half_annotation_dim});

  p_fIH = model.add_parameters({final_hidden_dim, embedding_dim + 2 * half_annotation_dim + output_state_dim});
  p_fHb = model.add_parameters({final_hidden_dim});
  p_fHO = model.add_parameters({tgt_vocab_size, final_hidden_dim});
  p_fOb = model.add_parameters({tgt_vocab_size});
}

vector<VariableIndex> AttentionalModel::BuildForwardAnnotations(const vector<WordId>& sentence, ComputationGraph& hg) {
  forward_builder.new_graph(hg);
  forward_builder.start_new_sequence();
  vector<VariableIndex> forward_annotations(sentence.size());
  for (unsigned t = 0; t < sentence.size(); ++t) {
    VariableIndex i_x_t = hg.add_lookup(p_Es, sentence[t]);
    VariableIndex i_y_t = forward_builder.add_input(Expression(&hg, i_x_t)).i;
    forward_annotations[t] = i_y_t;
  }
  return forward_annotations;
}

vector<VariableIndex> AttentionalModel::BuildReverseAnnotations(const vector<WordId>& sentence, ComputationGraph& hg) {
  reverse_builder.new_graph(hg);
  reverse_builder.start_new_sequence();
  vector<VariableIndex> reverse_annotations(sentence.size());
  for (unsigned t = sentence.size(); t > 0; ) {
    t--;
    VariableIndex i_x_t = hg.add_lookup(p_Es, sentence[t]);
    VariableIndex i_y_t = reverse_builder.add_input(Expression(&hg, i_x_t)).i;
    reverse_annotations[t] = i_y_t;
  }
  return reverse_annotations;
}

vector<VariableIndex> AttentionalModel::BuildAnnotationVectors(const vector<VariableIndex>& forward_annotations, const vector<VariableIndex>& reverse_annotations, ComputationGraph& hg) {
  vector<VariableIndex> annotations(forward_annotations.size());
  for (unsigned t = 0; t < forward_annotations.size(); ++t) {
    const VariableIndex& i_f = forward_annotations[t];
    const VariableIndex& i_r = reverse_annotations[t];
    VariableIndex i_h = hg.add_function<Concatenate>({i_f, i_r});
    annotations[t] = i_h;
  }
  return annotations;
}

OutputState AttentionalModel::GetNextOutputState(const VariableIndex& prev_context, const vector<VariableIndex>& annotations,
    const MLP& aligner, ComputationGraph& hg, vector<float>* out_alignment) {
  const unsigned source_size = annotations.size();

  VariableIndex new_state = output_builder.add_input(Expression(&hg, prev_context)).i;
  vector<VariableIndex> unnormalized_alignments(source_size); // e_ij

  for (unsigned s = 0; s < source_size; ++s) {
    double prior = 1.0;
    VariableIndex a_input = hg.add_function<Concatenate>({new_state, annotations[s]});
    VariableIndex a_hidden1 = hg.add_function<AffineTransform>({aligner.i_Hb, aligner.i_IH, a_input});
    VariableIndex a_hidden2 = hg.add_function<Tanh>({a_hidden1});
    VariableIndex a_output = hg.add_function<AffineTransform>({aligner.i_Ob, aligner.i_HO, a_hidden2});
    unnormalized_alignments[s] = a_output * prior;
  }

  VariableIndex unnormalized_alignment_vector = hg.add_function<Concatenate>(unnormalized_alignments);
  VariableIndex normalized_alignment_vector = hg.add_function<Softmax>({unnormalized_alignment_vector});
  if (out_alignment != NULL) {
    *out_alignment = as_vector(hg.forward());
  }
  VariableIndex annotation_matrix = hg.add_function<ConcatenateColumns>(annotations);
  VariableIndex context = hg.add_function<MatrixMultiply>({annotation_matrix, normalized_alignment_vector});

  OutputState os;
  os.state = new_state;
  os.context = context;
  return os;
}

VariableIndex AttentionalModel::ComputeOutputDistribution(const WordId prev_word, const VariableIndex state, const VariableIndex context, const MLP& final, ComputationGraph& hg) {
  VariableIndex prev_target_embedding = hg.add_lookup(p_Et, prev_word);
  VariableIndex final_input = hg.add_function<Concatenate>({prev_target_embedding, state, context});
  VariableIndex final_hidden1 = hg.add_function<AffineTransform>({final.i_Hb, final.i_IH, final_input}); 
  VariableIndex final_hidden2 = hg.add_function<Tanh>({final_hidden1});
  VariableIndex final_output = hg.add_function<AffineTransform>({final.i_Ob, final.i_HO, final_hidden2});
  return final_output;
}

vector<vector<float> > AttentionalModel::Align(const vector<WordId>& source, const vector<WordId>& target) {
  ComputationGraph hg;
  output_builder.new_graph(hg);
  output_builder.start_new_sequence();

  vector<VariableIndex> forward_annotations = BuildForwardAnnotations(source, hg);
  vector<VariableIndex> reverse_annotations = BuildReverseAnnotations(source, hg);
  vector<VariableIndex> annotations = BuildAnnotationVectors(forward_annotations, reverse_annotations, hg);

  VariableIndex i_aIH = hg.add_parameters(p_aIH);
  VariableIndex i_aHb = hg.add_parameters(p_aHb);
  VariableIndex i_aHO = hg.add_parameters(p_aHO);
  VariableIndex i_aOb = hg.add_parameters(p_aOb);
  MLP aligner = {i_aIH, i_aHb, i_aHO, i_aOb};

  VariableIndex i_fIH = hg.add_parameters(p_fIH);
  VariableIndex i_fHb = hg.add_parameters(p_fHb);
  VariableIndex i_fHO = hg.add_parameters(p_fHO);
  VariableIndex i_fOb = hg.add_parameters(p_fOb);
  MLP final = {i_fIH, i_fHb, i_fHO, i_fOb};

  VariableIndex i_bs = hg.add_parameters(p_bs);
  VariableIndex i_Ws = hg.add_parameters(p_Ws);

  vector<VariableIndex> output_states(target.size());
  vector<VariableIndex> contexts(target.size());

  VariableIndex zeroth_context_untransformed = hg.add_function<AffineTransform>({i_bs, i_Ws, reverse_annotations[0]});
  VariableIndex zeroth_context = hg.add_function<Tanh>({zeroth_context_untransformed});
  VariableIndex prev_context = zeroth_context;

  vector<vector<float> > alignment;
  for (unsigned t = 1; t < target.size() + 1; ++t) {
    vector<float> a;
    OutputState os = GetNextOutputState(prev_context, annotations, aligner, hg, &a);
    prev_context = os.context;
    alignment.push_back(a);
  }
  return alignment;
}

vector<WordId> AttentionalModel::Translate(const vector<WordId>& source, WordId kSOS, WordId kEOS, unsigned beam_size, unsigned max_length) {
  KBestList<vector<WordId> > kbest = TranslateKBest(source, kSOS, kEOS, 1, beam_size, max_length);
  return kbest.hypothesis_list().begin()->second;
}

KBestList<vector<WordId> > AttentionalModel::TranslateKBest(const vector<WordId>& source, WordId kSOS, WordId kEOS, unsigned k, unsigned beam_size, unsigned max_length) {
  ComputationGraph hg;
  output_builder.new_graph(hg);

  vector<VariableIndex> forward_annotations = BuildForwardAnnotations(source, hg);
  vector<VariableIndex> reverse_annotations = BuildReverseAnnotations(source, hg);
  vector<VariableIndex> annotations = BuildAnnotationVectors(forward_annotations, reverse_annotations, hg);

  VariableIndex i_aIH = hg.add_parameters(p_aIH);
  VariableIndex i_aHb = hg.add_parameters(p_aHb);
  VariableIndex i_aHO = hg.add_parameters(p_aHO);
  VariableIndex i_aOb = hg.add_parameters(p_aOb);
  MLP aligner = {i_aIH, i_aHb, i_aHO, i_aOb};

  VariableIndex i_fIH = hg.add_parameters(p_fIH);
  VariableIndex i_fHb = hg.add_parameters(p_fHb);
  VariableIndex i_fHO = hg.add_parameters(p_fHO);
  VariableIndex i_fOb = hg.add_parameters(p_fOb);
  MLP final = {i_fIH, i_fHb, i_fHO, i_fOb};

  VariableIndex i_bs = hg.add_parameters(p_bs);
  VariableIndex i_Ws = hg.add_parameters(p_Ws);

  VariableIndex zeroth_context_untransformed = hg.add_function<AffineTransform>({i_bs, i_Ws, reverse_annotations[0]});
  VariableIndex zeroth_context = hg.add_function<Tanh>({zeroth_context_untransformed});
  /* Up until here this is all boiler plate */

  KBestList<vector<WordId> > completed_hyps(beam_size);
  KBestList<vector<WordId> > top_hyps(beam_size);
  top_hyps.add(0.0, {});

  // Invariant: each element in top_hyps should have a length of "length"
  for (unsigned length = 0; length < max_length; ++length) {
    KBestList<vector<WordId> > new_hyps(beam_size);
    for (auto scored_hyp : top_hyps.hypothesis_list()) {
      double score = scored_hyp.first;
      vector<WordId>& hyp = scored_hyp.second;
      assert (hyp.size() == length);

      // XXX: Rebuild the whole output state sequence
      output_builder.start_new_sequence(); 
      OutputState os = GetNextOutputState(zeroth_context, annotations, aligner, hg);
      for (WordId word : hyp) {
        assert (word != kEOS);
        os = GetNextOutputState(os.context, annotations, aligner, hg);
      } 

      // Compute, normalize, and log the output distribution
      WordId prev_word = (hyp.size() > 0) ? hyp[hyp.size() - 1] : kSOS;
      VariableIndex unnormalized_output_distribution = ComputeOutputDistribution(prev_word, os.state, os.context, final, hg);
      VariableIndex output_distribution = hg.add_function<Softmax>({unnormalized_output_distribution});
      VariableIndex log_output_distribution = hg.add_function<Log>({output_distribution});
      vector<float> dist = as_vector(hg.incremental_forward());

      // Take the K best-looking words
      KBestList<WordId> best_words(beam_size);
      for (unsigned i = 0; i < dist.size(); ++i) {
        best_words.add(dist[i], i);
      }

      // For each of those K words, add it to the current hypothesis, and add the
      // resulting hyp to our kbest list, unless the new word is </s>,
      // in which case we add the new hyp to the list of completed hyps.
      for (pair<double, WordId> p : best_words.hypothesis_list()) {
        double word_score = p.first;
        WordId word = p.second;
        OutputState new_state = GetNextOutputState(os.context, annotations, aligner, hg);
        output_builder.rewind_one_step();

        double new_score = score + word_score;
        vector<WordId> new_hyp = hyp;
        new_hyp.push_back(word);
        if (new_hyp.size() == max_length || word == kEOS) {
          completed_hyps.add(new_score, new_hyp);
        }
        else {
          new_hyps.add(new_score, new_hyp);
        }
      }
    }
    top_hyps = new_hyps;
  }
  return completed_hyps;
}

vector<WordId> AttentionalModel::SampleTranslation(const vector<WordId>& source, WordId kSOS, WordId kEOS, unsigned max_length) {
  ComputationGraph hg;
  output_builder.new_graph(hg);
  output_builder.start_new_sequence();

  vector<VariableIndex> forward_annotations = BuildForwardAnnotations(source, hg);
  vector<VariableIndex> reverse_annotations = BuildReverseAnnotations(source, hg);
  vector<VariableIndex> annotations = BuildAnnotationVectors(forward_annotations, reverse_annotations, hg);

  VariableIndex i_aIH = hg.add_parameters(p_aIH);
  VariableIndex i_aHb = hg.add_parameters(p_aHb);
  VariableIndex i_aHO = hg.add_parameters(p_aHO);
  VariableIndex i_aOb = hg.add_parameters(p_aOb);
  MLP aligner = {i_aIH, i_aHb, i_aHO, i_aOb};

  VariableIndex i_fIH = hg.add_parameters(p_fIH);
  VariableIndex i_fHb = hg.add_parameters(p_fHb);
  VariableIndex i_fHO = hg.add_parameters(p_fHO);
  VariableIndex i_fOb = hg.add_parameters(p_fOb);
  MLP final = {i_fIH, i_fHb, i_fHO, i_fOb};

  VariableIndex i_bs = hg.add_parameters(p_bs);
  VariableIndex i_Ws = hg.add_parameters(p_Ws);

  VariableIndex zeroth_context_untransformed = hg.add_function<AffineTransform>({i_bs, i_Ws, reverse_annotations[0]});
  VariableIndex zeroth_context = hg.add_function<Tanh>({zeroth_context_untransformed});

  VariableIndex prev_context = zeroth_context;
  vector<WordId> output;
  unsigned prev_word = kSOS;
  while (prev_word != kEOS && output.size() < max_length) {
    OutputState os = GetNextOutputState(prev_context, annotations, aligner, hg);
    VariableIndex log_output_distribution = ComputeOutputDistribution(prev_word, os.state, os.context, final, hg);
    VariableIndex output_distribution = hg.add_function<Softmax>({log_output_distribution});
    vector<float> dist = as_vector(hg.incremental_forward());
    double r = rand01();
    unsigned w = 0;
    while (true) {
      r -= dist[w];
      if (r < 0.0) {
        break;
      }
      ++w;
    }
    output.push_back(w);
    prev_word = w;
    prev_context = os.context;
  }
  return output;
}

VariableIndex AttentionalModel::BuildGraph(const vector<WordId>& source, const vector<WordId>& target, ComputationGraph& hg) {
  // Target should always contain at least <s> and </s>
  assert (target.size() > 2);
  output_builder.new_graph(hg);
  output_builder.start_new_sequence();

  vector<VariableIndex> forward_annotations = BuildForwardAnnotations(source, hg);
  vector<VariableIndex> reverse_annotations = BuildReverseAnnotations(source, hg);
  vector<VariableIndex> annotations = BuildAnnotationVectors(forward_annotations, reverse_annotations, hg);

  VariableIndex i_aIH = hg.add_parameters(p_aIH);
  VariableIndex i_aHb = hg.add_parameters(p_aHb);
  VariableIndex i_aHO = hg.add_parameters(p_aHO);
  VariableIndex i_aOb = hg.add_parameters(p_aOb);
  MLP aligner = {i_aIH, i_aHb, i_aHO, i_aOb};

  VariableIndex i_fIH = hg.add_parameters(p_fIH);
  VariableIndex i_fHb = hg.add_parameters(p_fHb);
  VariableIndex i_fHO = hg.add_parameters(p_fHO);
  VariableIndex i_fOb = hg.add_parameters(p_fOb);
  MLP final = {i_fIH, i_fHb, i_fHO, i_fOb};

  VariableIndex i_bs = hg.add_parameters(p_bs);
  VariableIndex i_Ws = hg.add_parameters(p_Ws);

  vector<VariableIndex> output_states(target.size());
  vector<VariableIndex> contexts(target.size());

  // TODO: Verify that this crap corresponds to the comment below and is sane
  VariableIndex zeroth_context_untransformed = hg.add_function<AffineTransform>({i_bs, i_Ws, reverse_annotations[0]});
  contexts[0] = hg.add_function<Tanh>({zeroth_context_untransformed});

  for (unsigned t = 1; t < target.size(); ++t) {
    OutputState os = GetNextOutputState(contexts[t - 1], annotations, aligner, hg);
    output_states[t] = os.state;
    contexts[t] = os.context;
  }

  /*
  first input: W_s * h1_backwards
  afterwards, input: c_i
  c_i = sum_j(alpha_ij * h_j)
  alpha_ij = exp(e_ij) / sum_k(exp(e_ik))
  e_ij = a(s_i-1, h_j) where a is a FFNN
  */
  vector<VariableIndex> output_distributions(target.size() - 1);
  for (unsigned t = 1; t < target.size(); ++t) {
    WordId prev_word = target[t - 1];
    output_distributions[t - 1] = ComputeOutputDistribution(prev_word, output_states[t], contexts[t], final, hg);
  }

  vector<VariableIndex> errors(target.size() - 1);
  for (unsigned t = 1; t < target.size(); ++t) {
    VariableIndex output_distribution = output_distributions[t - 1];
    VariableIndex error = hg.add_function<PickNegLogSoftmax>({output_distribution}, target[t]);
    errors[t - 1] = error;
  }
  VariableIndex total_error = hg.add_function<Sum>(errors);
  return total_error;
}
